"""Train one model on BABILong in three curriculum stages, then test length extrapolation.

    python train.py --model transformer_mask --encoder_mask CCCCFFFF --output_dir runs/vanilla
    python train.py --model alibi --output_dir runs/alibi

The stages are ordered by how many facts an answer needs, and each adds tasks to the
ones before it, so the last stage trains on all 9:

    stage 1   one supporting fact        qa1 qa4 qa5 qa9 qa10
    stage 2   + two facts, or a tally    qa2 qa7 qa8
    stage 3   + three facts              qa3

The data comes straight from the Hugging Face Hub, loaded and fed to the DataLoader by
``load_dataset.py``: training
and per-epoch validation from ``RMT-team/babilong-train-5k-samples``, the evaluation
after a stage from the benchmark itself, ``RMT-team/babilong``. The training set covers
qa1-qa10 only, which is why the stages stop there; qa6 is left out as well (see
``STAGES``).

A sample is the bAbI facts hidden in PG19 text, followed by the question; the target is
the answer. Lengths are named as the benchmark names them (``0k`` is the bare facts).
Training mixes the samples of every length in ``--train_len`` (0k, 1k and 2k). After
every stage the model is evaluated at each of ``--eval_lens``, which goes on to 4k and
8k, lengths it never trained on. Writes ``stage<k>.pt`` after each stage and
``metrics.jsonl``. ``--start_stage 2`` continues from ``stage1.pt`` in the same folder.

With ``--mode online`` (or ``offline``) the run is tracked in Weights & Biases against
the optimizer step: ``train/loss`` and ``train/lr`` every ``--log_every`` steps,
``val/loss``, ``val/accuracy`` and ``val/accuracy/qa<N>`` every epoch, and
``eval/accuracy/len<L>`` (with ``eval/len<L>/qa<N>``) after every stage. The default
``--mode disabled`` logs nothing and needs no login.
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import wandb
from transformers import get_inverse_sqrt_schedule

from load_dataset import (LENGTHS, TRAIN_LENGTHS, length_tokens, load_babilong, load_tasks,
                          load_tokenizer, load_training_babilong, make_loader)
from tmodel.models import MODEL_TYPES, PE_TYPES, build_model

# The Hub's training set has qa1-qa10. Of those qa6 is left out, because kept tasks ask
# the same question over the same kind of story: "Is P in the L?" is asked by qa9 and
# qa10 too, which add negation / "either ... or".
STAGES = [
    [1, 4, 5, 9, 10],                     # one supporting fact
    [2, 7, 8],                            # two supporting facts, or a tally over several
    [3],                                  # three supporting facts
]

# Room for the question on top of the context when nothing may be truncated.
QUESTION_ROOM = 64


def forward(model, criterion, batch, device):
    src, tgt, tasks = batch
    src, tgt = src.to(device), tgt.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        logits = model(src, tgt[:, :-1])
    logits, tgt_out = logits.float(), tgt[:, 1:]
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
    return loss, logits, tgt_out, tasks


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    """Mean loss, and per task the share of answers with every token right (teacher-forced)."""
    model.eval()
    losses, right, total = [], {}, {}
    for batch in loader:
        loss, logits, tgt_out, tasks = forward(model, criterion, batch, device)
        losses.append(loss.item())
        ignore = tgt_out.eq(criterion.ignore_index)
        correct = (logits.argmax(-1).eq(tgt_out) | ignore).all(dim=1).cpu()
        for task, ok in zip(tasks.tolist(), correct.tolist()):
            right[task] = right.get(task, 0) + ok
            total[task] = total.get(task, 0) + 1
    model.train()
    return sum(losses) / len(losses), {f"qa{t}": right[t] / total[t] for t in sorted(total)}


def evaluate_lengths(args, model, criterion, tokenizer, tasks, device):
    """Accuracy at each of ``--eval_lens``: ``{length: {"mean": .., "qa1": .., ...}}``."""
    results = {}
    for length in args.eval_lens:
        samples = load_tasks(load_babilong, tasks, [length],
                             lambda n: range(min(n, args.eval_samples_per_task)))
        # Nothing is truncated here: the source is as long as the sample is.
        loader = make_loader(samples, tokenizer, args.eval_batch_size, args.workers,
                             src_len=max(length_tokens(length), args.src_len) + QUESTION_ROOM)
        _, accuracy = evaluate(model, criterion, loader, device)
        mean = sum(accuracy.values()) / len(accuracy)
        results[length] = dict(mean=mean, **accuracy)
        seen = "trained" if length in args.train_len else "extrapolation"
        print(f"  length {length:>6} ({seen:>13}) | acc {mean:.3f} | "
              + " ".join(f"{t} {a:.2f}" for t, a in accuracy.items()), flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=MODEL_TYPES, default="transformer_mask")
    p.add_argument("--encoder_mask", default="B", help="transformer_mask only, e.g. CCCCFFFF")
    p.add_argument("--pe", default="none", choices=PE_TYPES, help="transformer_mask only")
    p.add_argument("--train_len", nargs="+", default=["0k", "1k", "2k"], choices=TRAIN_LENGTHS,
                   help="training mixes the Hub's samples of these lengths")
    p.add_argument("--eval_lens", nargs="+", default=["0k", "1k", "2k", "4k", "8k"], choices=LENGTHS,
                   help="lengths evaluated after every stage; those not in --train_len are extrapolation")
    p.add_argument("--src_len", type=int, default=2048, help="longest training source")
    p.add_argument("--tgt_len", type=int, default=16)
    p.add_argument("--samples_per_task", type=int, default=None,
                   help="per task and length; default: all that validation leaves")
    p.add_argument("--val_samples_per_task", type=int, default=100,
                   help="per-epoch validation: held out from the end of each training file")
    p.add_argument("--eval_samples_per_task", type=int, default=100, help="per length, after a stage")
    p.add_argument("--epochs", type=int, nargs=3, default=[4, 4, 6], help="epochs per stage")
    p.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=2,
                   help="small: at 8k the per-head attention masks take ~2 GB per sample")
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--warmup_steps", type=int, default=4000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--output_dir", required=True)
    g = p.add_argument_group("wandb")
    g.add_argument("--project", default="Babilong_nopos")
    g.add_argument("--entity", default=None, help="wandb team or user; default is yours.")
    g.add_argument("--group", default=None)
    g.add_argument("--run_name", default=None, help="Default: the output folder's name.")
    g.add_argument("--mode", default="disabled", choices=["online", "offline", "disabled"],
                   help="offline writes the run locally for a later `wandb sync`.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_tokenizer()
    tokenizer.save_pretrained(args.output_dir)
    vocab_size, pad_id = len(tokenizer), tokenizer.pad_token_id

    # max_len only reaches roformer: its position table has to cover the longest evaluation.
    longest = max(args.src_len, *map(length_tokens, args.eval_lens)) + QUESTION_ROOM
    model = build_model(args.model, vocab_size, pad_id, mask_spec=args.encoder_mask, pe=args.pe,
                        max_len=longest + args.tgt_len).to(device)
    print(model.description)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    # One schedule over the whole run: it is stepped per optimizer step and not reset by a stage.
    scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=args.warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    step = 0
    if args.start_stage > 1:
        ckpt = torch.load(os.path.join(args.output_dir, f"stage{args.start_stage - 1}.pt"),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    # mode="disabled" gives a no-op run, so nothing below depends on the mode. A run
    # continued with --start_stage is a new wandb run that carries on at the same step.
    run = wandb.init(project=args.project, entity=args.entity, group=args.group,
                     name=args.run_name or os.path.basename(os.path.normpath(args.output_dir)),
                     mode=args.mode, job_type="train",
                     config=dict(vars(args), stages=STAGES, description=model.description))
    for stage in range(args.start_stage, len(STAGES) + 1):
        tasks = sorted(t for s in STAGES[:stage] for t in s)
        # Validated at the training lengths, on the last rows of each training file.
        n_val, n_train = args.val_samples_per_task, args.samples_per_task
        train = load_tasks(load_training_babilong, tasks, args.train_len,
                           lambda n: range(min(n - n_val, n_train or n)))
        val = load_tasks(load_training_babilong, tasks, args.train_len, lambda n: range(n - n_val, n))
        train_loader = make_loader(train, tokenizer, args.batch_size, args.workers, args.src_len,
                                   shuffle=True)
        val_loader = make_loader(val, tokenizer, args.batch_size, args.workers, args.src_len)
        print(f"=== stage {stage}: tasks {tasks} | {len(train_loader.dataset)} train samples, "
              f"{len(val_loader.dataset)} val, {args.epochs[stage - 1]} epochs ===", flush=True)

        for epoch in range(1, args.epochs[stage - 1] + 1):
            running = []
            for i, batch in enumerate(train_loader, 1):
                lr = scheduler.get_last_lr()[0]
                loss = forward(model, criterion, batch, device)[0]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                step += 1
                running.append(loss.item())
                if not math.isfinite(running[-1]):
                    raise RuntimeError(f"non-finite loss at stage {stage} epoch {epoch} step {i}")
                if i % args.log_every == 0:
                    recent = running[-args.log_every:]
                    print(f"  stage {stage} epoch {epoch} step {i}/{len(train_loader)} | "
                          f"loss {sum(recent) / len(recent):.4f}", flush=True)
                    run.log({"train/loss": sum(recent) / len(recent), "train/lr": lr,
                             "stage": stage, "epoch": epoch}, step=step)

            val_loss, accuracy = evaluate(model, criterion, val_loader, device)
            mean_acc = sum(accuracy.values()) / len(accuracy)
            print(f"stage {stage} epoch {epoch} | train {sum(running) / len(running):.4f} | "
                  f"val {val_loss:.4f} | acc {mean_acc:.3f} | "
                  + " ".join(f"{t} {a:.2f}" for t, a in accuracy.items()), flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(dict(stage=stage, epoch=epoch, step=step,
                                        train_loss=sum(running) / len(running), val_loss=val_loss,
                                        val_accuracy=mean_acc, **accuracy)) + "\n")
            run.log({"train/epoch_loss": sum(running) / len(running), "val/loss": val_loss,
                     "val/accuracy": mean_acc, "stage": stage, "epoch": epoch,
                     **{f"val/accuracy/{t}": a for t, a in accuracy.items()}}, step=step)

        # Saved before the length evaluation, so a failure at 8k cannot lose the stage.
        # The same layout as the RULER checkpoints, plus what a later stage needs to continue.
        out = os.path.join(args.output_dir, f"stage{stage}.pt")
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), step=step,
                        stage=stage, tasks=tasks, args=vars(args), vocab_size=vocab_size,
                        pad_token_id=pad_id, bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id), out + ".tmp")
        os.replace(out + ".tmp", out)
        print(f"saved {out}")

        print(f"stage {stage} accuracy by length:")
        by_length = evaluate_lengths(args, model, criterion, tokenizer, tasks, device)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(dict(stage=stage, step=step, train_len=args.train_len,
                                    accuracy_by_length=by_length)) + "\n")
        by_length_log = {}
        for length, result in by_length.items():
            by_length_log[f"eval/accuracy/len{length}"] = result["mean"]
            by_length_log.update({f"eval/len{length}/{t}": a for t, a in result.items() if t != "mean"})
        run.log(by_length_log, step=step)
    run.finish()


if __name__ == "__main__":
    main()
