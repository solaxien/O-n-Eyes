// Budgeted local retry and candidate selection. Everything here runs on the
// device; nothing is sent to a server. Pure functions so they can be tested.

export const POLICIES = Object.freeze({
  preview: { key: "preview", maxCandidates: 1, verify: false, retryOnFail: false, budgetMs: 15000 },
  eco: { key: "eco", maxCandidates: 1, verify: false, retryOnFail: false, budgetMs: 20000 },
  balanced: { key: "balanced", maxCandidates: 2, verify: true, retryOnFail: true, budgetMs: 45000 },
  quality: { key: "quality", maxCandidates: 3, verify: true, retryOnFail: false, budgetMs: 180000 },
  // The sampling reference stays a single, unselected image so it remains a
  // fair baseline.
  baseline: { key: "baseline", maxCandidates: 1, verify: false, retryOnFail: false, budgetMs: 60000 },
});

/**
 * Policy for the profile that actually ran. Preview, Eco and the reference
 * keep one candidate and only pixel checks; Balanced and Quality verify.
 */
export function selectionPolicy(profileKey, { verifyRequested = true, budgetMs } = {}) {
  const base = POLICIES[profileKey] || POLICIES.eco;
  const verify = verifyRequested && base.verify;
  return {
    ...base,
    verify,
    // Retries and multi-candidate search need a verifier to mean anything.
    maxCandidates: verify ? base.maxCandidates : 1,
    retryOnFail: verify && base.retryOnFail,
    budgetMs: Number.isFinite(budgetMs) && budgetMs > 0 ? budgetMs : base.budgetMs,
  };
}

/** Deterministic, distinct seed for candidate `index` (0 = user's seed). */
export function candidateSeed(seed, index) {
  return index === 0 ? seed >>> 0 : (seed + Math.imul(index, 1000003)) >>> 0;
}

/** Higher is better. Unverified candidates rank on pixel health only. */
export function candidateScore(candidate) {
  const summary = candidate.verifier?.summary,
    semantic = summary && summary.checked ? summary.score : null,
    pixel = candidate.cv?.usable === false ? 0 : 1,
    health = (candidate.cv?.quality?.after?.score ?? 50) / 100;
  return (semantic ?? 0) * 10 + pixel * 2 + health;
}

export function pickBest(candidates) {
  let best = 0;
  for (let i = 1; i < candidates.length; i++)
    if (candidateScore(candidates[i]) > candidateScore(candidates[best])) best = i;
  return best;
}

/**
 * Decide whether to generate another candidate.
 * Returns { next: boolean, reason: string }.
 */
export function shouldContinue(policy, candidates, elapsedMs) {
  const count = candidates.length,
    last = candidates[count - 1];
  if (count >= policy.maxCandidates)
    return { next: false, reason: "Candidate limit reached." };
  if (!policy.verify)
    return { next: false, reason: "No verifier in this mode, so no semantic retry." };
  if (!last?.verifier || last.verifier.unavailable)
    return { next: false, reason: "Verifier unavailable; semantic retry was not attempted." };
  const summary = last.verifier.summary;
  if (!summary.checked)
    return { next: false, reason: "Nothing in this prompt is verifiable on this device." };
  if (policy.retryOnFail && !summary.critical_failed)
    return { next: false, reason: "All checkable clauses passed." };
  if (!policy.retryOnFail && candidates.some((c) => c.verifier?.summary?.failed === 0))
    return { next: false, reason: "A candidate already passed every checkable clause." };
  const perCandidate = elapsedMs / count;
  if (elapsedMs + perCandidate > policy.budgetMs)
    return { next: false, reason: "Time budget would be exceeded; returning best effort." };
  return {
    next: true,
    reason: policy.retryOnFail
      ? "A checkable clause failed; retrying once with a new seed."
      : "Searching for a better candidate within the time budget.",
  };
}
