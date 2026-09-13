/**
 * MOCK pipeline response for issue #30.
 *
 * This module stands in for the backend pipeline (intent classifier +
 * historical retrieval + reply generator + auto/escalate decision policy,
 * see docs/specs-v0.md sections 4-7) while the backend is built.
 *
 * Issue #31 will replace `getMockedPipelineResult` with a real fetch to the
 * backend. The `PipelineResult` shape below is intentionally compatible with
 * the future backend JSON: { intent, confidence, evidence, reply, decision,
 * reason }.
 */

export type PipelineDecision = "auto" | "escalate";

export interface PipelineResult {
  /** Predicted support intent id (see docs/intent-taxonomy.md). */
  intent: string;
  /** Classifier confidence in [0, 1]. */
  confidence: number;
  /** Historical Case ids used as evidence for the reply. */
  evidence: string[];
  /** Drafted brand reply, grounded in the evidence. */
  reply: string;
  /** Auto-handle vs escalate verdict. */
  decision: PipelineDecision;
  /** One-line reason for the decision (specs-v0.md section 7). */
  reason: string;
}

/** Upper bound enforced by the form; mirrors a reasonable API limit. */
export const MAX_MESSAGE_LENGTH = 1000;

interface MockRule {
  match: RegExp;
  result: PipelineResult;
}

const MOCK_RULES: MockRule[] = [
  {
    match: /(hack|hacked|password|login|log ?in|sign ?in|locked out|credentials)/i,
    result: {
      intent: "account",
      confidence: 0.93,
      evidence: ["case_2779503", "case_1819027"],
      reply:
        "Thanks for flagging this — I can see this looks like an account access issue. " +
        "For your safety, please reset your password from the login page and sign out of all devices. " +
        "A human specialist will follow up here to confirm your account is secure.",
      decision: "escalate",
      reason:
        "High-risk category (security/credentials) — always escalates regardless of confidence.",
    },
  },
  {
    match: /(charg|payment|refund|billing|invoice|card|paypal|gift card|receipt)/i,
    result: {
      intent: "billing_payment",
      confidence: 0.88,
      evidence: ["case_367624", "case_2647464"],
      reply:
        "Sorry about the billing trouble. Similar past cases were resolved by checking the receipt " +
        "on the account page and retrying the payment method. A human specialist will review the " +
        "charge details with you here.",
      decision: "escalate",
      reason:
        "Payment dispute — requires account-specific investigation before auto-handling.",
    },
  },
  {
    match: /(play|song|shuffle|queue|skip|volume|error.*play|can't play)/i,
    result: {
      intent: "playback",
      confidence: 0.87,
      evidence: ["case_2920760", "case_1840273"],
      reply:
        "Sorry your music is not playing. Try a quick restart of the app, then check for updates — " +
        "that resolved similar playback cases. If the error persists, let us know what device you are on.",
      decision: "auto",
      reason:
        "High intent confidence with strong historical evidence; no high-risk signals.",
    },
  },
  {
    match: /(presale|pre-sale|presell|concert.*code|tour.*code)/i,
    result: {
      intent: "presale_codes",
      confidence: 0.95,
      evidence: ["case_2150438", "case_2203840"],
      reply:
        "Presale codes are sent to the email on your account before the sale window. " +
        "Check spam/promotions and the concert page — that is where similar cases found the code.",
      decision: "auto",
      reason:
        "High intent confidence with strong historical evidence; no high-risk signals.",
    },
  },
  {
    match: /(family|invite|member.*plan|join.*family)/i,
    result: {
      intent: "family_plan",
      confidence: 0.84,
      evidence: ["case_2978154", "case_1846413"],
      reply:
        "Sorry the Family invite is failing. Ask the plan owner to resend the invite link and make sure " +
        "everyone uses the same country on their accounts — that fixed similar cases.",
      decision: "auto",
      reason:
        "High intent confidence with strong historical evidence; no high-risk signals.",
    },
  },
];

const FALLBACK_MOCK: PipelineResult = {
  intent: "app_technical",
  confidence: 0.62,
  evidence: ["case_1516306", "case_2395935"],
  reply:
    "Thanks for reaching out — this looks like it may be an app issue. Try updating to the latest " +
    "version and restarting your device. A human specialist will take a look to confirm the next step.",
  decision: "escalate",
  reason:
    "Intent confidence is below the auto-handle threshold, so this needs human review.",
};

/**
 * Return a deterministic mocked pipeline result for any customer message.
 * Pure function of the message text (keyword rules + fallback), so the UI
 * is demoable offline with no backend.
 */
export function getMockedPipelineResult(customerMessage: string): PipelineResult {
  const message = customerMessage.trim();
  for (const rule of MOCK_RULES) {
    if (rule.match.test(message)) {
      return { ...rule.result, evidence: [...rule.result.evidence] };
    }
  }
  return { ...FALLBACK_MOCK, evidence: [...FALLBACK_MOCK.evidence] };
}
