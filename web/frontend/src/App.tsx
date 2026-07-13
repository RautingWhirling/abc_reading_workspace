import {
  CheckCircle2,
  Download,
  FileJson,
  Image as ImageIcon,
  Loader2,
  Play,
  RefreshCcw,
  Upload,
  XCircle
} from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type TabKey = "text" | "image" | "eval";
type RunState = "queued" | "running" | "completed" | "failed";

type RunStatus = {
  run_id: string;
  mode: TabKey;
  state: RunState;
  created_at: string;
  updated_at: string;
  total: number;
  completed: number;
  failed: number;
  current_event_id?: string | null;
  output_dir: string;
  trace_dir: string;
  results: RunResult[];
  errors: RunError[];
};

type RunResult = {
  event_id: string;
  event_name: string;
  selected_digital_human_ids: string[];
  content_llm?: Record<string, unknown>;
  json_output_path: string;
};

type RunError = {
  event_id?: string | null;
  message: string;
};

type EventForm = {
  event_id: string;
  domain: string;
  event_title: string;
  event_summary: string;
  target: string;
  opinion_variants: string;
};

type RunOptions = {
  profile_limit?: number | null;
  max_selected_nodes: number;
  risk_level?: "low" | "medium" | "high" | null;
  campaign_window_hours: number;
  max_frequency_per_day: number;
  allowed_platforms: string[];
  use_llm: boolean;
  event_limit?: number | null;
  event_id?: string | null;
};

const fallbackTemplate = {
  event_id: "web_event_001",
  domain: "general",
  event_title: "请输入事件标题",
  event_summary: "请输入事件概要",
  target: "引导公众理解事件影响并进行理性讨论。",
  is_synthetic: false,
  opinion_variants: ["请输入可用于发帖内容生成的观点变体。"]
};

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("text");
  const [inputMode, setInputMode] = useState<"form" | "json">("form");
  const [form, setForm] = useState<EventForm>(eventToForm(fallbackTemplate));
  const [rawJson, setRawJson] = useState(JSON.stringify(fallbackTemplate, null, 2));
  const [jsonError, setJsonError] = useState("");
  const [useLlm, setUseLlm] = useState(true);
  const [disableControls, setDisableControls] = useState(false);
  const [profileLimit, setProfileLimit] = useState("");
  const [maxSelectedNodes, setMaxSelectedNodes] = useState(5);
  const [riskLevel, setRiskLevel] = useState("");
  const [campaignWindowHours, setCampaignWindowHours] = useState(24);
  const [maxFrequencyPerDay, setMaxFrequencyPerDay] = useState(3);
  const [allowedPlatforms, setAllowedPlatforms] = useState("weibo_simulated");
  const [eventLimit, setEventLimit] = useState(200);
  const [eventIdFilter, setEventIdFilter] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState("");
  const [run, setRun] = useState<RunStatus | null>(null);
  const [selectedEventId, setSelectedEventId] = useState("");
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [trace, setTrace] = useState<Record<string, unknown> | null>(null);
  const [notice, setNotice] = useState("");
  const [loadingDetail, setLoadingDetail] = useState(false);

  useEffect(() => {
    fetchJson("/api/event-template")
      .then((payload) => {
        setForm(eventToForm(payload));
        setRawJson(JSON.stringify(payload, null, 2));
      })
      .catch(() => {
        setNotice("后端未就绪，已加载本地模板。");
      });
  }, []);

  useEffect(() => {
    if (!run?.run_id || run.state === "completed" || run.state === "failed") {
      return;
    }
    const timer = window.setInterval(() => {
      refreshRun(run.run_id);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [run?.run_id, run?.state]);

  useEffect(() => {
    if (run?.results?.length && !selectedEventId) {
      const first = run.results[0];
      setSelectedEventId(first.event_id);
      loadDetail(run.run_id, first.event_id);
    }
  }, [run?.results, run?.run_id, selectedEventId]);

  const options = useMemo<RunOptions>(() => {
    const platforms = allowedPlatforms
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    return {
      profile_limit: profileLimit.trim() ? Number(profileLimit) : null,
      max_selected_nodes: maxSelectedNodes,
      risk_level: riskLevel ? (riskLevel as RunOptions["risk_level"]) : null,
      campaign_window_hours: campaignWindowHours,
      max_frequency_per_day: maxFrequencyPerDay,
      allowed_platforms: platforms.length ? platforms : ["weibo_simulated"],
      use_llm: useLlm,
      event_limit: activeTab === "eval" ? eventLimit : null,
      event_id: eventIdFilter.trim() || null
    };
  }, [
    activeTab,
    allowedPlatforms,
    campaignWindowHours,
    eventIdFilter,
    eventLimit,
    maxFrequencyPerDay,
    maxSelectedNodes,
    profileLimit,
    riskLevel,
    useLlm
  ]);

  const isRunning = run?.state === "queued" || run?.state === "running" || disableControls;
  const progress = run && run.total > 0 ? Math.round(((run.completed + run.failed) / run.total) * 100) : 0;

  async function refreshRun(runId: string) {
    const payload = (await fetchJson(`/api/runs/${runId}`)) as RunStatus;
    setRun(payload);
  }

  async function startTextRun(event: FormEvent) {
    event.preventDefault();
    setDisableControls(true);
    setJsonError("");
    setNotice("");
    setDetail(null);
    setTrace(null);
    setSelectedEventId("");
    try {
      const hotEvent = inputMode === "json" ? parseRawEvent(rawJson) : formToEvent(form);
      const payload = (await postJson("/api/runs/text", {
        event: hotEvent,
        options
      })) as RunStatus;
      setRun(payload);
    } catch (error) {
      setJsonError(errorMessage(error));
    } finally {
      setDisableControls(false);
    }
  }

  async function startImageRun(event: FormEvent) {
    event.preventDefault();
    if (!imageFile) {
      setNotice("请选择一张图片。");
      return;
    }
    setDisableControls(true);
    setNotice("");
    setDetail(null);
    setTrace(null);
    setSelectedEventId("");
    try {
      const data = new FormData();
      data.append("image", imageFile);
      data.append("options", JSON.stringify(options));
      const response = await fetch("/api/runs/image", {
        method: "POST",
        body: data
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      setRun((await response.json()) as RunStatus);
    } catch (error) {
      setNotice(errorMessage(error));
    } finally {
      setDisableControls(false);
    }
  }

  async function startEvalRun() {
    setDisableControls(true);
    setNotice("");
    setDetail(null);
    setTrace(null);
    setSelectedEventId("");
    try {
      const payload = (await postJson("/api/eval-runs", { options })) as RunStatus;
      setRun(payload);
    } catch (error) {
      setNotice(errorMessage(error));
    } finally {
      setDisableControls(false);
    }
  }

  async function loadDetail(runId: string, eventId: string) {
    setLoadingDetail(true);
    setTrace(null);
    try {
      const payload = (await fetchJson(`/api/runs/${runId}/results/${eventId}`)) as Record<string, unknown>;
      setDetail(payload);
      const tracePayload = (await fetchJson(`/api/runs/${runId}/traces/${eventId}`)) as Record<string, unknown>;
      setTrace(tracePayload);
    } catch (error) {
      setNotice(errorMessage(error));
    } finally {
      setLoadingDetail(false);
    }
  }

  function handleImageChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setImageFile(file);
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }
    setImagePreview(file ? URL.createObjectURL(file) : "");
  }

  function updateForm<K extends keyof EventForm>(key: K, value: EventForm[K]) {
    const next = { ...form, [key]: value };
    setForm(next);
    if (inputMode === "form") {
      setRawJson(JSON.stringify(formToEvent(next), null, 2));
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>影响力事件分发控制台</h1>
          <p>当前输出目录：outputs/web_runs</p>
        </div>
        <div className="topbar-actions">
          <label className="toggle">
            <input type="checkbox" checked={useLlm} onChange={(event) => setUseLlm(event.target.checked)} />
            <span>{useLlm ? "LLM 开启" : "LLM 关闭"}</span>
          </label>
          <button className="icon-button" onClick={() => run?.run_id && refreshRun(run.run_id)} disabled={!run?.run_id}>
            <RefreshCcw size={18} />
            刷新
          </button>
        </div>
      </header>

      <main className="workspace-grid">
        <section className="panel input-panel">
          <div className="tabs" role="tablist" aria-label="输入模式">
            <button className={activeTab === "text" ? "tab active" : "tab"} onClick={() => setActiveTab("text")}>
              <FileJson size={17} />
              文本输入
            </button>
            <button className={activeTab === "image" ? "tab active" : "tab"} onClick={() => setActiveTab("image")}>
              <ImageIcon size={17} />
              图像输入
            </button>
            <button className={activeTab === "eval" ? "tab active" : "tab"} onClick={() => setActiveTab("eval")}>
              <Play size={17} />
              200 条 Eval
            </button>
          </div>

          {activeTab === "text" && (
            <form className="stack" onSubmit={startTextRun}>
              <div className="segmented">
                <button type="button" className={inputMode === "form" ? "active" : ""} onClick={() => setInputMode("form")}>
                  表单
                </button>
                <button type="button" className={inputMode === "json" ? "active" : ""} onClick={() => setInputMode("json")}>
                  JSON
                </button>
              </div>

              {inputMode === "form" ? (
                <div className="form-grid">
                  <label>
                    事件 ID
                    <input value={form.event_id} onChange={(event) => updateForm("event_id", event.target.value)} />
                  </label>
                  <label>
                    领域
                    <input value={form.domain} onChange={(event) => updateForm("domain", event.target.value)} />
                  </label>
                  <label className="wide">
                    标题
                    <input value={form.event_title} onChange={(event) => updateForm("event_title", event.target.value)} />
                  </label>
                  <label className="wide">
                    概要
                    <textarea value={form.event_summary} rows={4} onChange={(event) => updateForm("event_summary", event.target.value)} />
                  </label>
                  <label className="wide">
                    传播目标
                    <textarea value={form.target} rows={3} onChange={(event) => updateForm("target", event.target.value)} />
                  </label>
                  <label className="wide">
                    观点变体
                    <textarea value={form.opinion_variants} rows={6} onChange={(event) => updateForm("opinion_variants", event.target.value)} />
                  </label>
                </div>
              ) : (
                <label className="json-editor">
                  事件 JSON
                  <textarea value={rawJson} rows={18} onChange={(event) => setRawJson(event.target.value)} spellCheck={false} />
                </label>
              )}

              <OptionsPanel
                profileLimit={profileLimit}
                setProfileLimit={setProfileLimit}
                maxSelectedNodes={maxSelectedNodes}
                setMaxSelectedNodes={setMaxSelectedNodes}
                riskLevel={riskLevel}
                setRiskLevel={setRiskLevel}
                campaignWindowHours={campaignWindowHours}
                setCampaignWindowHours={setCampaignWindowHours}
                maxFrequencyPerDay={maxFrequencyPerDay}
                setMaxFrequencyPerDay={setMaxFrequencyPerDay}
                allowedPlatforms={allowedPlatforms}
                setAllowedPlatforms={setAllowedPlatforms}
              />

              {jsonError && <div className="message error">{jsonError}</div>}
              <button className="primary-button" type="submit" disabled={isRunning}>
                {isRunning ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
                运行单事件
              </button>
            </form>
          )}

          {activeTab === "image" && (
            <form className="stack" onSubmit={startImageRun}>
              <label className="upload-box">
                <Upload size={22} />
                <span>{imageFile ? imageFile.name : "选择图片"}</span>
                <input type="file" accept=".png,.jpg,.jpeg,.webp" onChange={handleImageChange} />
              </label>
              {imagePreview && (
                <div className="image-preview">
                  <img src={imagePreview} alt="事件图片预览" />
                </div>
              )}
              <OptionsPanel
                profileLimit={profileLimit}
                setProfileLimit={setProfileLimit}
                maxSelectedNodes={maxSelectedNodes}
                setMaxSelectedNodes={setMaxSelectedNodes}
                riskLevel={riskLevel}
                setRiskLevel={setRiskLevel}
                campaignWindowHours={campaignWindowHours}
                setCampaignWindowHours={setCampaignWindowHours}
                maxFrequencyPerDay={maxFrequencyPerDay}
                setMaxFrequencyPerDay={setMaxFrequencyPerDay}
                allowedPlatforms={allowedPlatforms}
                setAllowedPlatforms={setAllowedPlatforms}
              />
              <button className="primary-button" type="submit" disabled={isRunning || !imageFile}>
                {isRunning ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
                识别并运行
              </button>
            </form>
          )}

          {activeTab === "eval" && (
            <div className="stack">
              <div className="form-grid">
                <label>
                  事件数量
                  <input type="number" min={1} max={200} value={eventLimit} onChange={(event) => setEventLimit(Number(event.target.value))} />
                </label>
                <label>
                  指定事件 ID
                  <input value={eventIdFilter} onChange={(event) => setEventIdFilter(event.target.value)} placeholder="留空则跑前 200 条" />
                </label>
              </div>
              <OptionsPanel
                profileLimit={profileLimit}
                setProfileLimit={setProfileLimit}
                maxSelectedNodes={maxSelectedNodes}
                setMaxSelectedNodes={setMaxSelectedNodes}
                riskLevel={riskLevel}
                setRiskLevel={setRiskLevel}
                campaignWindowHours={campaignWindowHours}
                setCampaignWindowHours={setCampaignWindowHours}
                maxFrequencyPerDay={maxFrequencyPerDay}
                setMaxFrequencyPerDay={setMaxFrequencyPerDay}
                allowedPlatforms={allowedPlatforms}
                setAllowedPlatforms={setAllowedPlatforms}
              />
              <button className="primary-button" type="button" onClick={startEvalRun} disabled={isRunning}>
                {isRunning ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
                一键运行 Eval
              </button>
            </div>
          )}
        </section>

        <section className="panel status-panel">
          <SectionTitle title="任务状态" />
          {notice && <div className="message">{notice}</div>}
          {run ? (
            <>
              <div className="status-row">
                <StatusBadge state={run.state} />
                <span className="run-id">{run.run_id}</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <div className="stat-grid">
                <Stat label="完成" value={run.completed} />
                <Stat label="失败" value={run.failed} />
                <Stat label="总数" value={run.total} />
                <Stat label="进度" value={`${progress}%`} />
              </div>
              <div className="path-list">
                <div>
                  <strong>输出</strong>
                  <span>{run.output_dir}</span>
                </div>
                <div>
                  <strong>Trace</strong>
                  <span>{run.trace_dir}</span>
                </div>
                {run.current_event_id && (
                  <div>
                    <strong>当前</strong>
                    <span>{run.current_event_id}</span>
                  </div>
                )}
              </div>
              {run.errors.length > 0 && (
                <div className="error-list">
                  {run.errors.slice(-4).map((item, index) => (
                    <div key={`${item.event_id}-${index}`} className="message error">
                      {item.event_id || "任务"}：{item.message}
                    </div>
                  ))}
                </div>
              )}
              <div className="result-list">
                {run.results.map((item) => (
                  <button
                    key={item.event_id}
                    className={selectedEventId === item.event_id ? "result-item active" : "result-item"}
                    onClick={() => {
                      setSelectedEventId(item.event_id);
                      loadDetail(run.run_id, item.event_id);
                    }}
                  >
                    <span>{item.event_id}</span>
                    <strong>{item.event_name}</strong>
                    <small>{item.selected_digital_human_ids.length} 个数字人</small>
                  </button>
                ))}
              </div>
            </>
          ) : (
            <div className="empty-state">暂无任务</div>
          )}
        </section>

        <section className="panel result-panel">
          <div className="result-head">
            <SectionTitle title="策略结果" />
            {run && selectedEventId && (
              <a className="icon-button link-button" href={`/api/runs/${run.run_id}/files/${selectedEventId}`} target="_blank" rel="noreferrer">
                <Download size={18} />
                JSON
              </a>
            )}
          </div>
          {loadingDetail && <div className="empty-state">加载中</div>}
          {!loadingDetail && detail && <ResultDetail payload={detail} trace={trace} />}
          {!loadingDetail && !detail && <div className="empty-state">选择一个结果查看详情</div>}
        </section>
      </main>
    </div>
  );
}

type OptionsPanelProps = {
  profileLimit: string;
  setProfileLimit: (value: string) => void;
  maxSelectedNodes: number;
  setMaxSelectedNodes: (value: number) => void;
  riskLevel: string;
  setRiskLevel: (value: string) => void;
  campaignWindowHours: number;
  setCampaignWindowHours: (value: number) => void;
  maxFrequencyPerDay: number;
  setMaxFrequencyPerDay: (value: number) => void;
  allowedPlatforms: string;
  setAllowedPlatforms: (value: string) => void;
};

function OptionsPanel(props: OptionsPanelProps) {
  return (
    <div className="options-panel">
      <SectionTitle title="运行参数" />
      <div className="form-grid">
        <label>
          主选数量
          <input type="number" min={1} max={20} value={props.maxSelectedNodes} onChange={(event) => props.setMaxSelectedNodes(Number(event.target.value))} />
        </label>
        <label>
          风险等级
          <select value={props.riskLevel} onChange={(event) => props.setRiskLevel(event.target.value)}>
            <option value="">自动</option>
            <option value="low">low</option>
            <option value="medium">medium</option>
            <option value="high">high</option>
          </select>
        </label>
        <label>
          传播窗口
          <input type="number" min={1} max={168} value={props.campaignWindowHours} onChange={(event) => props.setCampaignWindowHours(Number(event.target.value))} />
        </label>
        <label>
          频率上限
          <input type="number" min={1} max={24} value={props.maxFrequencyPerDay} onChange={(event) => props.setMaxFrequencyPerDay(Number(event.target.value))} />
        </label>
        <label>
          Profile 限制
          <input value={props.profileLimit} onChange={(event) => props.setProfileLimit(event.target.value)} placeholder="留空加载全部" />
        </label>
        <label>
          平台
          <input value={props.allowedPlatforms} onChange={(event) => props.setAllowedPlatforms(event.target.value)} />
        </label>
      </div>
    </div>
  );
}

function ResultDetail({ payload, trace }: { payload: Record<string, unknown>; trace: Record<string, unknown> | null }) {
  const strategy = payload["五维调度策略"] as Record<string, unknown> | undefined;
  const target = strategy?.["目标对象"] as Record<string, unknown> | undefined;
  const time = strategy?.["时间"] as Record<string, unknown> | undefined;
  const frequency = strategy?.["频率"] as Record<string, unknown> | undefined;
  const platform = strategy?.["平台"] as Record<string, unknown> | undefined;
  const content = strategy?.["内容"] as Record<string, unknown> | undefined;
  const selectedIds = (target?.["选取数字人id组"] as string[] | undefined) ?? [];
  const traceFiles = ((trace?.files as Array<{ name: string; payload: unknown }> | undefined) ?? []).slice(0, 8);

  return (
    <div className="detail-stack">
      <div className="summary-band">
        <div>
          <span>事件</span>
          <strong>{String(payload["事件名称"] ?? "-")}</strong>
        </div>
        <div>
          <span>格式</span>
          <strong>{String(payload["输出格式版本"] ?? "-")}</strong>
        </div>
      </div>

      <section className="result-section">
        <h3>目标对象</h3>
        <p>{String(target?.["传播目标"] ?? "-")}</p>
        <div className="id-row">
          {selectedIds.map((id) => (
            <span key={id}>{id}</span>
          ))}
        </div>
        <DataTable items={(target?.["数字人分工"] as Array<Record<string, unknown>> | undefined) ?? []} />
      </section>

      <section className="result-section">
        <h3>时间与频率</h3>
        <div className="split-grid">
          <DataTable items={(time?.["数字人时间安排"] as Array<Record<string, unknown>> | undefined) ?? []} />
          <DataTable items={(frequency?.["数字人频率"] as Array<Record<string, unknown>> | undefined) ?? []} />
        </div>
      </section>

      <section className="result-section">
        <h3>平台</h3>
        <p>{String(platform?.["平台模式"] ?? "-")}</p>
        <DataTable items={(platform?.["数字人平台分发"] as Array<Record<string, unknown>> | undefined) ?? []} />
      </section>

      <section className="result-section">
        <h3>内容</h3>
        <DataTable items={(content?.["数字人发帖内容"] as Array<Record<string, unknown>> | undefined) ?? []} />
      </section>

      <section className="result-section">
        <h3>动作清单</h3>
        <DataTable items={(content?.["动作清单"] as Array<Record<string, unknown>> | undefined) ?? []} />
      </section>

      {traceFiles.length > 0 && (
        <section className="result-section">
          <h3>Trace</h3>
          <div className="trace-list">
            {traceFiles.map((file) => (
              <details key={file.name}>
                <summary>{file.name}</summary>
                <pre>{JSON.stringify(file.payload, null, 2)}</pre>
              </details>
            ))}
          </div>
        </section>
      )}

      <section className="result-section">
        <h3>原始 JSON</h3>
        <pre>{JSON.stringify(payload, null, 2)}</pre>
      </section>
    </div>
  );
}

function DataTable({ items }: { items: Array<Record<string, unknown>> }) {
  if (!items.length) {
    return <div className="empty-table">无数据</div>;
  }
  const keys = Array.from(new Set(items.flatMap((item) => Object.keys(item)))).slice(0, 6);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {keys.map((key) => (
              <th key={key}>{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((item, index) => (
            <tr key={index}>
              {keys.map((key) => (
                <td key={key}>{renderCell(item[key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusBadge({ state }: { state: RunState }) {
  const icon = state === "completed" ? <CheckCircle2 size={18} /> : state === "failed" ? <XCircle size={18} /> : <Loader2 className="spin" size={18} />;
  return <span className={`status-badge ${state}`}>{icon}{state}</span>;
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SectionTitle({ title }: { title: string }) {
  return <h2>{title}</h2>;
}

function eventToForm(event: Record<string, unknown>): EventForm {
  return {
    event_id: String(event.event_id ?? ""),
    domain: String(event.domain ?? "general"),
    event_title: String(event.event_title ?? ""),
    event_summary: String(event.event_summary ?? ""),
    target: String(event.target ?? ""),
    opinion_variants: Array.isArray(event.opinion_variants)
      ? event.opinion_variants.map(String).join("\n")
      : String(event.opinion_variants ?? "")
  };
}

function formToEvent(form: EventForm) {
  return {
    event_id: form.event_id,
    domain: form.domain,
    event_title: form.event_title,
    event_summary: form.event_summary,
    target: form.target,
    is_synthetic: false,
    opinion_variants: form.opinion_variants
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean)
  };
}

function parseRawEvent(raw: string) {
  const payload = JSON.parse(raw);
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("事件 JSON 必须是对象。");
  }
  return payload;
}

async function fetchJson(path: string) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

async function postJson(path: string, body: unknown) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

async function readError(response: Response) {
  try {
    const payload = await response.json();
    return typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload);
  } catch {
    return response.statusText;
  }
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function renderCell(value: unknown) {
  if (Array.isArray(value)) {
    return value.join("、");
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value ?? "");
}

export default App;
