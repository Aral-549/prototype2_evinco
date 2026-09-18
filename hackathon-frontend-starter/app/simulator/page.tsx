"use client";

import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  Surface,
  SectionHeader,
  RiskChip,
  Chip,
  Stat,
  Separator,
  Meter,
  NotMeasured,
} from "@/components/ui/primitives";
import {
  crore,
  probability,
  sourceLabel,
  featureLabel,
  TIER_TEXT,
  TIER_LABEL,
} from "@/lib/format";
import { api, Assessment, HorizonResponse, HorizonEntry } from "@/lib/api";
import { SiteHeader } from "@/components/site-header";
import { AlertCircle, AlertTriangle, Play, Calendar, Activity, Info, FileText } from "lucide-react";

interface PresetProject {
  name: string;
  sector: string;
  agency: string;
  origCost: number;
  expenditure: number;
  progress: number;
  elapsed: number;
  daysStale: number;
  dispute: boolean;
  remarks: string;
}

const PRESETS: Record<string, PresetProject> = {
  'MOSPI-RLW-0101': { name: 'USBRL Kashmir Rail Link (Northern Railway)', sector: 'Railways', agency: 'Northern Railway', origCost: 37012, expenditure: 15800, progress: 37.2, elapsed: 96, daysStale: 42, dispute: true, remarks: 'RoW in Ramban-Banihal section pending with forest clearance Stage-2. High-altitude tunnel boring halted due to geological surprises.' },
  'MOSPI-DFC-0042': { name: 'Western Dedicated Freight Corridor (DFCCIL)', sector: 'Railways', agency: 'DFCCIL', origCost: 28181, expenditure: 24000, progress: 72.5, elapsed: 54, daysStale: 15, dispute: false, remarks: 'Land RoW pending in 2 districts with forest clearance Stage-2 under MoEF review.' },
  'MOSPI-MTR-0312': { name: 'Mumbai Metro Line 3 (MMRCL)', sector: 'Urban Development', agency: 'MMRCL', origCost: 33405, expenditure: 22100, progress: 68.4, elapsed: 78, daysStale: 8, dispute: false, remarks: 'Station box completion delayed at Marol Naka due to utility shifting. TBM breakthrough expected Q2.' },
  'MOSPI-PWR-0881': { name: 'Barh Super Thermal Power Station (NTPC)', sector: 'Power', agency: 'NTPC', origCost: 21556, expenditure: 18900, progress: 85.0, elapsed: 108, daysStale: 60, dispute: true, remarks: 'Contractor arbitration pending since 2021. Ash disposal site clearance from State PCB awaited.' },
  'MOSPI-HGY-0504': { name: 'AIIMS Bilaspur Healthcare Campus', sector: 'Urban Development', agency: 'HLL Lifecare', origCost: 1471, expenditure: 490, progress: 42.0, elapsed: 36, daysStale: 5, dispute: false, remarks: '' },
};

export default function SimulatorPage() {
  const [selectedPreset, setSelectedPreset] = useState<string>('MOSPI-DFC-0042');
  const [formState, setFormState] = useState<PresetProject>(PRESETS['MOSPI-DFC-0042']);
  
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [horizons, setHorizons] = useState<HorizonResponse | null>(null);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const p = e.target.value;
    setSelectedPreset(p);
    if (PRESETS[p]) {
      setFormState(PRESETS[p]);
    }
  };

  const evaluate = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        project_id: 'SIM-EXP-001',
        project_name: formState.name || 'Custom Project',
        sector: formState.sector,
        implementing_agency: formState.agency,
        original_cost: formState.origCost,
        expenditure: formState.expenditure,
        physical_progress: formState.progress,
        original_duration_months: 60.0,
        months_elapsed: formState.elapsed,
        current_delay_months: Math.max(0, formState.elapsed - 60),
        days_since_last_update: formState.daysStale,
        contractual_dispute_active: formState.dispute,
        remarks: formState.remarks,
      };

      const [ass, hor] = await Promise.all([
        api.assess(payload),
        api.horizons(payload)
      ]);
      setAssessment(ass);
      setHorizons(hor);
    } catch (err: any) {
      setError(err.message || 'Failed to evaluate project');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    evaluate();
  }, []); // Run once on mount

  return (
    <>
      <SiteHeader />
      <main className="mx-auto w-full max-w-5xl px-4 pb-16 pt-6 sm:px-6">
        <header className="mb-8">
          <h1 className="t-large-title">What-If Simulator</h1>
          <p className="t-body mt-2 max-w-prose text-fg-secondary">
            Adjust project parameters to see real-time risk predictions and slip forecasts.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Form */}
          <div className="lg:col-span-5 space-y-6">
            <Surface className="p-6">
              <div className="space-y-4">
                <div>
                  <label className="block t-section mb-1">Project Preset</label>
                  <select 
                    value={selectedPreset} 
                    onChange={handlePresetChange}
                    className="w-full rounded-[var(--radius-card)] bg-surface-sunken ring-1 ring-hairline p-2 t-subhead"
                  >
                    {Object.entries(PRESETS).map(([id, preset]) => (
                      <option key={id} value={id}>{id}: {preset.name}</option>
                    ))}
                  </select>
                </div>

                <Separator />

                <div className="space-y-3">
                  <div>
                    <label className="block t-section mb-1">Sector</label>
                    <select 
                      value={formState.sector} 
                      onChange={e => setFormState({...formState, sector: e.target.value})}
                      className="w-full rounded-[var(--radius-card)] bg-surface-sunken ring-1 ring-hairline p-2 t-subhead"
                    >
                      <option>Railways</option>
                      <option>Road Transport & Highways</option>
                      <option>Power</option>
                      <option>Petroleum</option>
                      <option>Urban Development</option>
                      <option>Civil Aviation</option>
                    </select>
                  </div>

                  <div>
                    <label className="block t-section mb-1">Implementing Agency</label>
                    <input 
                      type="text" 
                      value={formState.agency} 
                      onChange={e => setFormState({...formState, agency: e.target.value})}
                      className="w-full rounded-[var(--radius-card)] bg-surface-sunken ring-1 ring-hairline p-2 t-subhead"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block t-section mb-1">Sanction ₹ Cr</label>
                      <input 
                        type="number" 
                        value={formState.origCost} 
                        onChange={e => setFormState({...formState, origCost: parseFloat(e.target.value) || 0})}
                        className="w-full rounded-[var(--radius-card)] bg-surface-sunken ring-1 ring-hairline p-2 t-subhead tabular"
                      />
                    </div>
                    <div>
                      <label className="block t-section mb-1">Outlay ₹ Cr</label>
                      <input 
                        type="number" 
                        value={formState.expenditure} 
                        onChange={e => setFormState({...formState, expenditure: parseFloat(e.target.value) || 0})}
                        className="w-full rounded-[var(--radius-card)] bg-surface-sunken ring-1 ring-hairline p-2 t-subhead tabular"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="flex justify-between t-section mb-1">
                      <span>Physical Progress</span>
                      <span className="tabular">{formState.progress}%</span>
                    </label>
                    <input 
                      type="range" 
                      min="0" max="100" step="0.5" 
                      value={formState.progress} 
                      onChange={e => setFormState({...formState, progress: parseFloat(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="flex justify-between t-section mb-1">
                      <span>Elapsed Duration</span>
                      <span className="tabular">{formState.elapsed} / 60 mos ({Math.round(formState.elapsed / 60 * 100)}%)</span>
                    </label>
                    <input 
                      type="range" 
                      min="6" max="120" step="1" 
                      value={formState.elapsed} 
                      onChange={e => setFormState({...formState, elapsed: parseInt(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="flex justify-between t-section mb-1">
                      <span>Stale Days</span>
                      <span className="tabular">{formState.daysStale} d</span>
                    </label>
                    <input 
                      type="range" 
                      min="0" max="120" step="1" 
                      value={formState.daysStale} 
                      onChange={e => setFormState({...formState, daysStale: parseInt(e.target.value) || 0})}
                      className="w-full"
                    />
                  </div>

                  <div className="flex items-center space-x-2 pt-2">
                    <input 
                      type="checkbox" 
                      id="dispute"
                      checked={formState.dispute} 
                      onChange={e => setFormState({...formState, dispute: e.target.checked})}
                      className="rounded bg-surface-sunken ring-1 ring-hairline"
                    />
                    <label htmlFor="dispute" className="t-section cursor-pointer">Active Legal Dispute</label>
                  </div>

                  <div>
                    <label className="block t-section mb-1">Delay Remarks</label>
                    <textarea 
                      value={formState.remarks} 
                      onChange={e => setFormState({...formState, remarks: e.target.value})}
                      className="w-full rounded-[var(--radius-card)] bg-surface-sunken ring-1 ring-hairline p-2 t-subhead h-24 resize-none"
                    />
                  </div>

                  <button 
                    onClick={evaluate}
                    disabled={loading}
                    className="w-full flex items-center justify-center space-x-2 bg-fg text-bg rounded-[var(--radius-card)] p-3 font-semibold disabled:opacity-50 transition-opacity"
                  >
                    <Play className="w-4 h-4" />
                    <span>{loading ? 'Evaluating...' : 'Evaluate'}</span>
                  </button>
                </div>
              </div>
            </Surface>
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-7 space-y-6">
            {error ? (
              <Surface className="p-8 text-center text-critical">
                <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="t-body">{error}</p>
              </Surface>
            ) : loading || !assessment || !horizons ? (
              <div className="space-y-6 animate-pulse">
                <div className="h-48 rounded-[var(--radius-card)] bg-surface-sunken" />
                <div className="h-40 rounded-[var(--radius-card)] bg-surface-sunken" />
                <div className="h-40 rounded-[var(--radius-card)] bg-surface-sunken" />
              </div>
            ) : (
              <>
                {/* Hero Assessment Card */}
                <Surface className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h2 className="t-title-3 mb-2">{assessment.project_name}</h2>
                      <div className="flex gap-2">
                        <RiskChip tier={assessment.risk_tier} />
                        <Chip>{assessment.sector}</Chip>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="t-footnote text-fg-secondary">GovScore</div>
                      <div className={cn("t-large-title tabular font-semibold", TIER_TEXT[assessment.risk_tier])}>
                        {assessment.gov_score.toFixed(1)}
                      </div>
                      <div className="t-caption text-fg-tertiary">
                        {sourceLabel(assessment.dominant_source)}
                      </div>
                    </div>
                  </div>
                  
                  <Separator className="my-4" />
                  
                  <div className="grid grid-cols-2 gap-4">
                    <Stat 
                      label="Capital-at-Risk"
                      value={crore(assessment.capital_at_risk_crores)}
                      className={assessment.capital_at_risk_crores > 0 ? "text-critical" : ""}
                    />
                    <div className="space-y-1">
                      <div className="t-caption text-fg-secondary">Score Breakdown</div>
                      <div className="t-footnote tabular">
                        Base: {(assessment.p_model * 100).toFixed(1)} <br/>
                        Floor: {assessment.rule_floor.toFixed(1)} <br/>
                        <span className="text-fg-tertiary">Max(100·P, Floor) applied</span>
                      </div>
                    </div>
                  </div>
                </Surface>

                {/* Slip Forecast */}
                <Surface className="p-6">
                  <h3 className="t-headline flex items-center gap-2 mb-4">
                    <Activity className="w-4 h-4 text-fg-tertiary" />
                    Slip Forecast
                  </h3>
                  <div className="space-y-4">
                    {['1m', '3m', '6m'].map(horizon => {
                      const data = horizons.horizons[horizon];
                      if (!data) return null;
                      return (
                        <div key={horizon} className="space-y-1.5">
                          <div className="flex justify-between t-subhead">
                            <span>{horizon} Horizon</span>
                            <span className="tabular font-medium">{probability(data.probability)}</span>
                          </div>
                          <Meter value={data.probability * 100} tone={data.probability >= 0.75 ? "bg-critical" : data.probability >= 0.5 ? "bg-high" : data.probability >= 0.25 ? "bg-moderate" : "bg-low"} />
                        </div>
                      );
                    })}
                  </div>
                </Surface>

                {/* Statutory Flags */}
                <Surface className="p-6">
                  <h3 className="t-headline flex items-center gap-2 mb-4">
                    <AlertTriangle className="w-4 h-4 text-fg-tertiary" />
                    Statutory Flags
                  </h3>
                  <div className="space-y-2">
                    {assessment.rule_signals.length > 0 ? assessment.rule_signals.map(rule => (
                      <div 
                        key={rule.flag_id}
                        className={cn(
                          "p-3 rounded-[var(--radius-card)] t-footnote flex items-start gap-3",
                          rule.is_active ? "bg-critical-wash border border-critical/10" : "bg-surface-sunken"
                        )}
                      >
                        {rule.is_active ? <AlertCircle className="w-4 h-4 text-critical shrink-0 mt-0.5" /> : <Info className="w-4 h-4 text-fg-tertiary shrink-0 mt-0.5" />}
                        <div>
                          <div className={cn("font-medium", rule.is_active ? "text-critical" : "text-fg-secondary")}>
                            {rule.rule_name}
                          </div>
                          <div className="text-fg-tertiary mt-0.5">{rule.statutory_rationale}</div>
                        </div>
                      </div>
                    )) : (
                      <div className="t-footnote text-fg-tertiary py-2 text-center">No statutory flags measured</div>
                    )}
                  </div>
                </Surface>

                {/* SHAP Drivers */}
                <Surface className="p-6">
                  <h3 className="t-headline flex items-center gap-2 mb-4">
                    <FileText className="w-4 h-4 text-fg-tertiary" />
                    Key Risk Drivers
                  </h3>
                  <div className="space-y-3">
                    {assessment.shap_drivers.slice(0, 5).map((driver, i) => {
                      const isPos = driver.shap_value > 0;
                      return (
                        <div key={i} className="flex items-center gap-3">
                          <div className="w-16 text-right tabular t-caption text-fg-tertiary">
                            {isPos ? '+' : ''}{driver.shap_value.toFixed(2)}
                          </div>
                          <div className="flex-1 relative h-6 rounded-[var(--radius-chip)] bg-surface-sunken overflow-hidden">
                            <div 
                              className={cn(
                                "absolute top-0 bottom-0", 
                                isPos ? "bg-critical right-1/2" : "bg-accent left-1/2"
                              )}
                              style={{ width: `${Math.min(Math.abs(driver.shap_value) * 10, 50)}%` }}
                            />
                          </div>
                          <div className="w-32 truncate t-caption" title={featureLabel(driver.feature_name)}>
                            {featureLabel(driver.feature_name)}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </Surface>

                {/* Recommended Actions */}
                {assessment.prescriptive_interventions?.length > 0 && (
                  <Surface className="p-6">
                    <h3 className="t-headline mb-4">Recommended Actions</h3>
                    <ul className="list-disc pl-5 space-y-2 t-subhead text-fg-secondary">
                      {assessment.prescriptive_interventions.map((intervention, i) => (
                        <li key={i}>{intervention}</li>
                      ))}
                    </ul>
                  </Surface>
                )}
              </>
            )}
          </div>
        </div>
      </main>
    </>
  );
}
