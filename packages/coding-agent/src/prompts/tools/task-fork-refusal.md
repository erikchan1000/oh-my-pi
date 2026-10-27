Task execution failed: `fork: true` refused. {{#if disabled}}The `task.forkEnabled` setting is off. Retry without `fork`, or enable the setting.
{{else if noSessionFile}}A fork branches the parent's persisted session, and this session has no session file. Retry without `fork`.
{{else if noAgentId}}This session has no agent id, so the live parent session to branch cannot be identified. Retry without `fork`.
{{else if noParent}}The live parent session to branch is not registered. Retry without `fork`.
{{else if noUsage}}The parent's context size could not be measured. Retry without `fork`.
{{else if tooLarge}}The parent context is {{tokens}} tokens, above `task.forkMaxParentTokens` ({{maxTokens}}). Every fork child re-sends the parent prefix at cache-read price on each turn. Retry without `fork` and write a self-contained brief, or compact the parent first.
{{else if missingState}}The parent's leaf entry, model or prompt cache key is missing, so the child cannot share its prompt cache. Retry without `fork`.
{{else if unsupportedTools}}The parent has tools a fork child cannot share byte for byte ({{tools}}): MCP, extension or custom tools. A fork child must reuse the parent's exact tool list to hit the prompt cache. Retry without `fork`.
{{else if missingTools}}The child session would not receive these parent tools: {{tools}} (gated on UI, task depth or settings). A fork child must reuse the parent's exact tool list to hit the prompt cache. Retry without `fork`.
{{else if toolDrift}}The child's tools [{{childTools}}] differ from the parent's [{{parentTools}}], so it cannot share the parent's prompt cache. Retry without `fork`.
{{else if legacyFork}}The persisted fork child `{{id}}` predates the recorded fork contract (no `forkOfParent` in its session_init), so its parent-identical tool list cannot be restored. It was not revived.
{{else if conflict}}`fork` cannot be combined with `{{field}}`: a fork child runs on the parent's model, thinking level and tools and returns its final message. Retry without `{{field}}`, or without `fork`.
{{/if}}
