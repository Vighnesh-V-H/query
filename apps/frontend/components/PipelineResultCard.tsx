import type { PipelineResult } from "../lib/mock";

interface Props {
  result: PipelineResult;
  customerMessage: string;
}

function confidencePercent(confidence: number): number {
  return Math.round(confidence * 100);
}

export default function PipelineResultCard({ result, customerMessage }: Props) {
  const isAuto = result.decision === "auto";
  const pct = confidencePercent(result.confidence);

  return (
    <section aria-live="polite" aria-label="Pipeline result" className="card">
      <div className="card-row card-row--split">
        <h2>Pipeline result</h2>
        <span
          className={isAuto ? "badge badge--auto" : "badge badge--escalate"}
          role="status"
          aria-label={isAuto ? "Decision: auto-handle" : "Decision: escalate"}
        >
          {isAuto ? "AUTO" : "ESCALATE"}
        </span>
      </div>

      <p className="muted quoted">&ldquo;{customerMessage}&rdquo;</p>

      <dl className="fields">
        <div className="field">
          <dt>Intent</dt>
          <dd>
            <code>{result.intent}</code>
          </dd>
        </div>

        <div className="field">
          <dt>
            Confidence <span className="muted">({pct}%)</span>
          </dt>
          <dd>
            <div
              className="meter"
              role="progressbar"
              aria-valuenow={pct}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`Intent confidence ${pct} percent`}
            >
              <div className="meter-fill" style={{ width: `${pct}%` }} />
            </div>
          </dd>
        </div>

        <div className="field">
          <dt>Reply</dt>
          <dd>{result.reply}</dd>
        </div>

        <div className="field">
          <dt>Evidence</dt>
          <dd>
            {result.evidence.length > 0 ? (
              <ul className="evidence">
                {result.evidence.map((id) => (
                  <li key={id}>
                    <code>{id}</code>
                  </li>
                ))}
              </ul>
            ) : (
              <span className="muted">No supporting Historical Cases found.</span>
            )}
          </dd>
        </div>

        <div className="field">
          <dt>Reason</dt>
          <dd>{result.reason}</dd>
        </div>
      </dl>
    </section>
  );
}
