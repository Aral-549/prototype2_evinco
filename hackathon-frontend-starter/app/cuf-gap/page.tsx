"use client"

import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"
import { Surface, SectionHeader, Stat, Separator, NotMeasured, Meter } from "@/components/ui/primitives"
import { api, NotMeasuredError, CufGap } from "@/lib/api"
import { SiteHeader } from "@/components/site-header"
import { percent } from "@/lib/format"
import { BarChart3, FlaskConical, FileText, AlertTriangle, Info, CheckCircle2 } from "lucide-react"

export default function CufGapPage() {
  const [data, setData] = useState<CufGap | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    async function load() {
      try {
        setLoading(true)
        const result = await api.cufGap()
        setData(result)
      } catch (err: any) {
        setError(err)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  return (
    <>
      <SiteHeader />
      <main className="mx-auto w-full max-w-5xl px-4 pb-16 pt-6 sm:px-6">
        <div className="space-y-6">
          <div className="space-y-1">
            <h1 className="t-large-title text-fg">CUF 2.0 Information Bound</h1>
            <p className="t-title-3 text-fg-secondary">What more could a better form capture?</p>
          </div>

          {loading ? (
            <div className="space-y-4">
               <div className="h-32 bg-surface-sunken animate-pulse rounded-[var(--radius-card)]" />
               <div className="h-32 bg-surface-sunken animate-pulse rounded-[var(--radius-card)]" />
            </div>
          ) : error ? (
            error instanceof NotMeasuredError || error.message.includes("503") ? (
              <Surface className="p-6">
                <NotMeasured title="Data not available at this time." />
              </Surface>
            ) : (
              <Surface className="p-6">
                <p className="t-body text-critical">Failed to load: {error.message}</p>
              </Surface>
            )
          ) : data && (
            <>
              {/* Observable Feature Ceiling Card */}
              <Surface className="p-6 space-y-4">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-fg-secondary" />
                  <h2 className="t-title-3 text-fg">Observable Feature Ceiling</h2>
                </div>
                <div className="space-y-1">
                  <div className="t-headline text-fg-secondary">Out-of-fold AUC ceiling</div>
                  <div className="t-large-title text-fg">{data.discriminative_auc_ceiling.toFixed(3)}</div>
                </div>
                <div className="pt-2 pb-1">
                  <Meter value={data.discriminative_auc_ceiling * 100} tone="bg-accent" className="" />
                </div>
                <p className="t-caption text-fg-tertiary">{data.observable_ceiling_provenance}</p>
              </Surface>

              {/* NLP Delay-Narrative Augmentation Card */}
              <Surface className="p-6 space-y-4">
                <div className="flex items-center gap-2">
                  <FlaskConical className="w-5 h-5 text-fg-secondary" />
                  <h2 className="t-title-3 text-fg">NLP Delay-Narrative Augmentation</h2>
                </div>
                
                {data.nlp_augmentation.auc_delta === -1.0 ? (
                  <div className="p-4 bg-moderate/10 ring-1 ring-moderate/20 rounded-[var(--radius-card)] space-y-2">
                    <div className="flex items-center gap-2 text-moderate font-medium t-subhead">
                      <AlertTriangle className="w-4 h-4" />
                      NOT MEASURABLE
                    </div>
                    <p className="t-subhead text-moderate">
                      The public CUF feed does not publish granular delay-narrative remarks. The NLP proxy extractors (contractor cashflow, litigation, environment/forest, land acquisition, law & order) cannot be evaluated until the remarks column is populated.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="t-headline text-emerald-600">+{data.nlp_augmentation.auc_delta.toFixed(2)} AUC</div>
                    <div className="flex flex-wrap gap-2">
                      {data.nlp_augmentation.proxy_names.map(name => (
                        <div key={name} className="px-3 py-1 rounded-full bg-surface-sunken ring-1 ring-hairline t-footnote text-fg-secondary">
                          {name}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </Surface>

              {/* Methodology Statement Card */}
              <Surface className="p-6 flex gap-4">
                <Info className="w-5 h-5 text-fg-secondary shrink-0" />
                <p className="t-subhead text-fg-secondary">{data.methodology_statement}</p>
              </Surface>

              <Separator />

              {/* CUF 2.0 Structured Field Proposals */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-fg-secondary" />
                  <h2 className="t-title-2 text-fg">CUF 2.0 Structured Field Proposals</h2>
                </div>

                {/* Proposal 1 */}
                <Surface className="p-6 space-y-3">
                  <h3 className="t-headline text-fg">% Right-of-Way (RoW) Unencumbered at Statutory Sanction Date</h3>
                  <p className="t-subhead text-fg-secondary">
                    Awarding contracts before achieving 80% encumbrance-free RoW produces systemic stalls. CAG Report No. 19 of 2023 found Bharatmala Phase-I suffered 60% of delays from late RoW.
                  </p>
                  <div className="flex gap-2 pt-2">
                    <span className="px-2 py-1 rounded-md bg-surface-sunken ring-1 ring-hairline t-caption text-fg-secondary">Input: Float (0.0–100.0)</span>
                    <span className="px-2 py-1 rounded-md bg-accent/10 text-accent t-caption font-medium ring-1 ring-accent/20">P1 MANDATORY</span>
                  </div>
                </Surface>

                {/* Proposal 2 */}
                <Surface className="p-6 space-y-3">
                  <h3 className="t-headline text-fg">Categorical Stage-Gate Clearances</h3>
                  <p className="t-subhead text-fg-secondary">
                    Forest, Wildlife, Railway crossing clearances tracked as binary gates (obtained / pending / not applicable) enable the model to learn clearance-queue effects.
                  </p>
                  <div className="flex gap-2 pt-2">
                    <span className="px-2 py-1 rounded-md bg-surface-sunken ring-1 ring-hairline t-caption text-fg-secondary">Input: Enum (OBTAINED / PENDING / NA)</span>
                    <span className="px-2 py-1 rounded-md bg-accent/10 text-accent t-caption font-medium ring-1 ring-accent/20">P1 MANDATORY</span>
                  </div>
                </Surface>

                {/* Proposal 3 */}
                <Surface className="p-6 space-y-3">
                  <h3 className="t-headline text-fg">Winning Bid to Departmental Estimate Ratio</h3>
                  <p className="t-subhead text-fg-secondary">
                    Aggressive low-ball bids (ratio &lt; 0.75) correlate with contractor distress and mid-project arbitration in NHAI and Railway corridor awards.
                  </p>
                  <div className="flex gap-2 pt-2">
                    <span className="px-2 py-1 rounded-md bg-surface-sunken ring-1 ring-hairline t-caption text-fg-secondary">Input: Float (0.0–2.0)</span>
                    <span className="px-2 py-1 rounded-md bg-surface-sunken ring-1 ring-hairline t-caption text-fg-secondary">P2 RECOMMENDED</span>
                  </div>
                </Surface>
              </div>
            </>
          )}
        </div>
      </main>
    </>
  )
}
