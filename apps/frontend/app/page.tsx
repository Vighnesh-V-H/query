"use client";

import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import PipelineResultCard from "../components/PipelineResultCard";
import {
  MAX_MESSAGE_LENGTH,
  getMockedPipelineResult,
  type PipelineResult,
} from "../lib/mock";

/** Simulated latency so loading/empty states are visible while mocked. */
const MOCK_DELAY_MS = 400;

export default function Home() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [submittedMessage, setSubmittedMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const timerRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    return () => {
      window.clearTimeout(timerRef.current);
    };
  }, []);

  function validate(value: string): string | null {
    if (value.trim().length === 0) {
      return "Please enter a customer message.";
    }
    if (value.length > MAX_MESSAGE_LENGTH) {
      return `Message is too long — keep it under ${MAX_MESSAGE_LENGTH} characters.`;
    }
    return null;
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const problem = validate(message);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    setIsLoading(true);
    window.clearTimeout(timerRef.current);
    const snapshot = message.trim();
    // Mocked async pipeline call — #31 replaces this with a backend fetch.
    timerRef.current = window.setTimeout(() => {
      setResult(getMockedPipelineResult(snapshot));
      setSubmittedMessage(snapshot);
      setIsLoading(false);
    }, MOCK_DELAY_MS);
  }

  function handleClear() {
    window.clearTimeout(timerRef.current);
    setMessage("");
    setResult(null);
    setSubmittedMessage("");
    setError(null);
    setIsLoading(false);
  }

  const remaining = MAX_MESSAGE_LENGTH - message.length;
  const countText =
    remaining >= 0 ? `${remaining} left` : `${-remaining} over the limit`;

  return (
    <main className="page">
      <header className="hero">
        <p className="eyebrow">SpotifyCares &middot; v0 demo</p>
        <h1>Query — support agent</h1>
        <p className="muted">
          Type a customer message, submit, and see the mocked pipeline output: intent,
          reply, evidence, and the auto/escalate decision.{" "}
          <span className="mock-tag">Mocked data — no backend call yet (#31).</span>
        </p>
      </header>

      <div className="stack">
        <section className="card" aria-label="Message form">
          <h2>Customer message</h2>
          <form onSubmit={handleSubmit} noValidate>
            <label htmlFor="customer-message" className="label">
              What does the customer need help with?
            </label>
            <textarea
              id="customer-message"
              name="message"
              rows={4}
              maxLength={MAX_MESSAGE_LENGTH + 100}
              placeholder="e.g. My music isn't playing — I keep getting an error"
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              aria-invalid={error ? true : undefined}
              aria-describedby="form-error char-count"
              disabled={isLoading}
            />
            <div className="form-meta">
              <span
                id="form-error"
                role={error ? "alert" : undefined}
                className={error ? "error" : "muted"}
              >
                {error ?? "Try: billing, login, playback, family, or presale."}
              </span>
              <span id="char-count" className="muted">
                {countText}
              </span>
            </div>
            <div className="actions">
              <button type="submit" disabled={isLoading}>
                {isLoading ? "Running pipeline…" : "Submit"}
              </button>
              <button
                type="button"
                className="secondary"
                onClick={handleClear}
                disabled={isLoading && result === null}
              >
                Clear
              </button>
            </div>
          </form>
        </section>

        {isLoading && (
          <section className="card" aria-label="Loading" aria-busy="true">
            <p className="muted">Running mocked pipeline…</p>
          </section>
        )}

        {!isLoading && result && (
          <PipelineResultCard result={result} customerMessage={submittedMessage} />
        )}

        {!isLoading && !result && !error && (
          <section className="card card--empty" aria-label="Empty state">
            <p className="muted">
              No result yet — submit a message to see the mocked pipeline output.
            </p>
          </section>
        )}
      </div>

      <footer className="footer muted">
        Frontend-only mock for issue #30. Backend integration lands in #31.
      </footer>
    </main>
  );
}
