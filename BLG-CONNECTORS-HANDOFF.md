# BLG connectors and Dreaming: where it stands

Status as of November 15, 2026. This note is for review of the work on this fork. It lives on its own branch and is not meant to be merged.

## What this work is for

Big Little Gyms (BLG) wants any agent that connects to Signet to answer questions about any client and act on BLG's SOPs and past decisions. That needs client data from Stripe, GoHighLevel and Gmail in Signet, tied to the right client.

## Branches and pull requests

All branches are on this fork. Pull requests are drafts against the fork, stacked in this order.

| Branch | PR | What it adds |
|---|---|---|
| `feat/stripe-source` | #1 | Read-only Stripe source provider |
| `feat/gohighlevel-source` | #3 | Read-only GoHighLevel source provider (contacts, opportunities) |
| `feat/gmail-source` | #2 | Read-only Gmail source provider |
| `feat/gohighlevel-conversations` | #4 | GoHighLevel conversations and messages; recall test for Stripe artifacts |
| `feat/source-identity-lookup` | none | `GET /api/sources/identity` and `signet_identity_lookup`: one person across sources by email, phone or ID |
| `feat/stripe-more-resources` | none | Opt-in Stripe refunds, disputes, credit notes, coupons, products, prices, payouts, balance transactions, events; bounded retry when the DB owner is busy |
| `fix/acpx-dreaming-pass-tools` | #5 | Fix for Dreaming over ACPX (see below). Based on `main`, independent of the connector stack |
| `local/oogie-backfill-overrides` | none | Record of two private edits in the deployed build. Not a proposal |

## What is deployed

The BLG machine runs a build of `feat/stripe-more-resources` plus the two edits on `local/oogie-backfill-overrides`. The workspace holds about 203,000 source artifacts:

- GoHighLevel: 28,697 contacts, 12,757 opportunities, 20,327 conversations (latest 300 messages each).
- Stripe: 2,259 customers, 2,109 subscriptions, 31,303 invoices, 39,181 charges, plus the nine opt-in types.
- Gmail: loaded from a file export through an Obsidian-style file source, not the Gmail provider. About 1,700 of roughly 52,000 files were indexed at last check, at about 2.4 files a minute.

Dreaming and pipeline v2 are off in that workspace.

## What the tests showed

**Recall on the deployed workspace.** Ordinary recall returned no connector records on 7 of 7 client questions. `hybridRecall` uses source artifacts only as fill when results are under the limit, and connector artifacts get no embeddings. Source-only search answered 2 of 7.

**Dreaming on an isolated workspace.** 115 records for three clients (Stripe, GoHighLevel, Gmail export, contract text), `executor: acpx`, `agent: claude`, Sonnet, with the fix on `fix/acpx-dreaming-pass-tools`:

- 108 of 115 artifacts consumed in 11 passes over 2 hours 45 minutes; 137 claims written.
- All three businesses got Stripe, GoHighLevel, Gmail and contract facts on the same entity. Stripe child records that carry only a customer ID were filed under the right business.
- 21 of 82 `apply_ontology_ops` calls were rejected, nearly all by the exact-quote check.
- 28.3 million tokens in total, about 260,000 per record, nearly all cached reads. One pass hit the 20-minute timeout and its work was lost.
- Haiku skipped the evidence step in 3 of 5 passes and is not usable for this.

## Bugs found in Signet

1. **Dreaming over ACPX never advances.** Fixed on `fix/acpx-dreaming-pass-tools` (PR #5), with a route test.
2. **ACPX failures have no detail.** The pass error is `exited 1:` because the real message is in the JSON on stdout.
3. **`tokensConsumed` is wrong for ACPX passes.** It shows the prompt estimate.
4. **Missing index for the file watcher.** The active-artifact-paths query timed out at about 200,000 artifacts. This index fixed it on the deployed workspace and should be a migration:
   `CREATE INDEX idx_memory_artifacts_native_paths ON memory_artifacts(agent_id, harness, source_kind, source_id, source_path, is_deleted)`
5. **Docs.** `executor: codex` is retired but still documented. A working ACPX Dreaming target also needs `permissions: approve-all`, which is not documented.
6. **Workspace isolation.** A new workspace imports the machine's Claude and Codex transcripts and native memories from the home directory. No config switch was found; the test had to run with a different `HOME`.

## Questions for review

1. **Bulk records and cost.** Is there an intended cheaper path for large sets of structured records than one Dreaming read per artifact? One option is for connectors to emit one profile document per client.
2. **Recall of source artifacts.** Should connector artifacts be embedded or ranked in ordinary recall, or is fill-only the intended contract?
3. **People and relations.** Dreaming created business entities but no person entities for the owners, wrote nothing to `relations`, and put nearly every claim under one aspect per business. Is that expected from the current prompt, or a model issue?
4. **Known-good Dreaming config.** What target configuration do you run Dreaming with?

## Known gaps in this fork's work

- `withOwnerBusyRetry` has no test that exercises the retry.
- Seven CLI tests failed in the run that added the extra Stripe types; whether they fail on `main` was not checked.
- Identity lookup caps (200 records, 3 expansion rounds) truncate for clients with long histories, and the lookup does not join Gmail From/To.
- GoHighLevel conversations longer than 300 messages are truncated.
- GoHighLevel public-API conversation paging can skip records that share a sort value.
- PR descriptions for #1–#4 predate the later findings above.

## Suggested review order

1. PR #5: small, independent, fixes a real defect on `main`.
2. PR #1: sets the provider pattern the others follow.
3. Questions 1 and 2 above, before reviewing the rest of the stack, since the answers may change the connector design.
