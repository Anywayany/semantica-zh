import { useCallback, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import { useDropzone } from "react-dropzone";
import { useTranslation } from "react-i18next";
import {
  AlertCircle,
  ArrowRight,
  Check,
  CheckCircle2,
  FileSearch,
  FileText,
  KeyRound,
  Loader2,
  Network,
  RotateCcw,
  Sparkles,
  Trash2,
  UploadCloud,
} from "lucide-react";

interface EntityCandidate {
  id: string;
  text: string;
  label: string;
  confidence: number;
  mention_count: number;
  properties: Record<string, unknown>;
}

interface RelationCandidate {
  id: string;
  source_id: string;
  target_id: string;
  predicate: string;
  confidence: number;
  context: string;
  mention_count: number;
  properties: Record<string, unknown>;
}

interface DocumentPreview {
  document_id: string;
  filename: string;
  media_type: string;
  parser: string;
  language: "en" | "zh";
  extraction_method: string;
  execution: {
    requested_method: ExtractionMethod;
    actual_methods: string[];
    provider: string | null;
    model: string | null;
    provider_available: boolean;
    fallback_used: boolean;
    ontology_profile: string;
  };
  ontology_profile: string;
  character_count: number;
  chunk_count: number;
  text_preview: string;
  entities: EntityCandidate[];
  relations: RelationCandidate[];
  warnings: string[];
}

interface CommitResult {
  status: "success";
  document_node_id: string;
  nodes_added: number;
  edges_added: number;
  entities_submitted: number;
  relations_submitted: number;
}

interface DocumentGraphBuilderProps {
  onCommitted?: () => void | Promise<void>;
  onOpenGraph?: () => void;
}

type ExtractionMethod = "auto" | "rules" | "llm";
type DocumentLanguage = "auto" | "zh" | "en";
type OntologyProfile = "general" | "business" | "biomedical" | "legal" | "custom";

interface ProviderCapability {
  id: string;
  available: boolean;
  default_model: string;
  reason: string;
  credential_source: "session" | "server" | "none" | "not_required";
  session_configured: boolean;
}

interface ExtractionCapabilities {
  providers: ProviderCapability[];
  local_nlp_available: boolean;
  ontology_profiles: Record<string, { entity_types: string[]; relation_types: string[] }>;
}

interface LLMExtractionErrorDetail {
  code: string;
  provider: string;
  model: string;
  phase: string;
  chunk: number;
  total_chunks: number;
  attempts: number;
  retryable: boolean;
  upstream_status: number | null;
}

const panelStyle: CSSProperties = {
  border: "1px solid var(--ws-border)",
  borderRadius: "var(--ws-radius)",
  background: "rgba(7, 12, 22, 0.7)",
};

const fieldStyle: CSSProperties = {
  width: "100%",
  minHeight: 38,
  borderRadius: 9,
  border: "1px solid var(--ws-border)",
  background: "rgba(0, 0, 0, 0.24)",
  color: "var(--ws-text)",
  padding: "8px 10px",
  fontSize: 12,
};

function responseError(payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && "message" in detail) {
      const message = (detail as { message?: unknown }).message;
      if (typeof message === "string") return message;
    }
  }
  return fallback;
}

export function DocumentGraphBuilder({ onCommitted, onOpenGraph }: DocumentGraphBuilderProps) {
  const { t, i18n } = useTranslation();
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState<DocumentLanguage>("auto");
  const [extractionMethod, setExtractionMethod] = useState<ExtractionMethod>("auto");
  const [provider, setProvider] = useState("openai");
  const [model, setModel] = useState("gpt-4o-mini");
  const [ontologyProfile, setOntologyProfile] = useState<OntologyProfile>("general");
  const [customEntityTypes, setCustomEntityTypes] = useState("PERSON, ORG, GPE");
  const [customRelationTypes, setCustomRelationTypes] = useState("works_for, located_in");
  const [capabilities, setCapabilities] = useState<ExtractionCapabilities | null>(null);
  const [minConfidence, setMinConfidence] = useState(0.45);
  const [preview, setPreview] = useState<DocumentPreview | null>(null);
  const [selectedEntityIds, setSelectedEntityIds] = useState<Set<string>>(new Set());
  const [selectedRelationIds, setSelectedRelationIds] = useState<Set<string>>(new Set());
  const [isExtracting, setIsExtracting] = useState(false);
  const [isCommitting, setIsCommitting] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [isUpdatingCredential, setIsUpdatingCredential] = useState(false);
  const [credentialError, setCredentialError] = useState("");
  const [credentialNotice, setCredentialNotice] = useState("");
  const [error, setError] = useState("");
  const [commitResult, setCommitResult] = useState<CommitResult | null>(null);

  const refreshCapabilities = useCallback(async () => {
    const response = await fetch("/api/documents/capabilities");
    if (!response.ok) throw new Error("Capabilities request failed");
    const nextCapabilities = await response.json() as ExtractionCapabilities;
    setCapabilities(nextCapabilities);
    return nextCapabilities;
  }, []);

  useEffect(() => {
    let cancelled = false;
    void fetch("/api/documents/capabilities")
      .then(async (response) => {
        if (!response.ok) throw new Error("Capabilities request failed");
        return response.json() as Promise<ExtractionCapabilities>;
      })
      .then((nextCapabilities) => {
        if (!cancelled) setCapabilities(nextCapabilities);
      })
      .catch(() => {
        if (!cancelled) setCapabilities(null);
      });
    return () => { cancelled = true; };
  }, []);

  const resetResult = useCallback(() => {
    setPreview(null);
    setSelectedEntityIds(new Set());
    setSelectedRelationIds(new Set());
    setCommitResult(null);
    setError("");
  }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const nextFile = acceptedFiles[0] ?? null;
    setFile(nextFile);
    resetResult();
  }, [resetResult]);

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      "text/plain": [".txt", ".md", ".markdown"],
      "text/html": [".html", ".htm"],
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    },
    maxFiles: 1,
    maxSize: 25 * 1024 * 1024,
  });

  const entityLabels = useMemo(
    () => new Map(preview?.entities.map((entity) => [entity.id, entity.text]) ?? []),
    [preview],
  );
  const effectiveRelationIds = useMemo(() => {
    if (!preview) return new Set<string>();
    return new Set(
      preview.relations
        .filter((relation) => (
          selectedRelationIds.has(relation.id)
          && selectedEntityIds.has(relation.source_id)
          && selectedEntityIds.has(relation.target_id)
        ))
        .map((relation) => relation.id),
    );
  }, [preview, selectedEntityIds, selectedRelationIds]);

  const currentStep = commitResult ? 3 : preview ? 2 : 1;
  const locale = i18n.resolvedLanguage?.startsWith("zh") ? "zh-CN" : "en-US";
  const providerOptions = capabilities?.providers ?? [];
  const selectedProvider = providerOptions.find((item) => item.id === provider) ?? null;
  const selectedProviderReason = selectedProvider?.reason.includes("API key is missing")
    ? t("documentBuilder.providerReasons.missingApiKey")
    : selectedProvider?.reason.includes("not installed")
      ? t("documentBuilder.providerReasons.missingDependency")
      : selectedProvider?.reason.includes("not reachable")
        ? t("documentBuilder.providerReasons.notReachable")
        : selectedProvider?.reason ?? t("documentBuilder.checkingProvider");

  function localizedWarning(warning: string): string {
    if (warning.includes("Automatic mode selected an offline extractor")) {
      return t("documentBuilder.warnings.autoFallback", { provider });
    }
    if (warning.includes("No entity candidates were extracted")) return t("documentBuilder.warnings.noEntities");
    if (warning.includes("no relation candidates were extracted")) return t("documentBuilder.warnings.noRelations");
    return warning;
  }

  function extractionResponseError(payload: unknown, fallback: string): string {
    if (!payload || typeof payload !== "object" || !("detail" in payload)) {
      return fallback;
    }
    const detail = (payload as { detail?: unknown }).detail;
    if (!detail || typeof detail !== "object" || !("code" in detail)) {
      return responseError(payload, fallback);
    }
    const failure = detail as LLMExtractionErrorDetail;
    if (failure.retryable) {
      return t("documentBuilder.errors.retryableProviderFailure", {
        provider: failure.provider,
        model: failure.model,
        chunk: failure.chunk,
        total: failure.total_chunks,
        attempts: failure.attempts,
      });
    }
    return t("documentBuilder.errors.providerRequestFailure", {
      provider: failure.provider,
      model: failure.model,
      code: failure.code,
      chunk: failure.chunk,
      total: failure.total_chunks,
    });
  }

  async function configureSessionCredential() {
    const submittedKey = apiKey.trim();
    if (!submittedKey) return;
    setIsUpdatingCredential(true);
    setCredentialError("");
    setCredentialNotice("");
    try {
      const response = await fetch("/api/documents/providers/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, api_key: submittedKey }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(responseError(payload, t("documentBuilder.credentials.errors.saveFailed")));
      }
      setApiKey("");
      setCredentialNotice(t("documentBuilder.credentials.saved"));
      await refreshCapabilities();
      resetResult();
    } catch (caught: unknown) {
      setCredentialError(caught instanceof Error ? caught.message : t("documentBuilder.credentials.errors.saveFailed"));
    } finally {
      setIsUpdatingCredential(false);
    }
  }

  async function clearSessionCredential() {
    setIsUpdatingCredential(true);
    setCredentialError("");
    setCredentialNotice("");
    try {
      const response = await fetch(`/api/documents/providers/session/${encodeURIComponent(provider)}`, {
        method: "DELETE",
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(responseError(payload, t("documentBuilder.credentials.errors.clearFailed")));
      }
      setApiKey("");
      setCredentialNotice(t("documentBuilder.credentials.cleared"));
      await refreshCapabilities();
      resetResult();
    } catch (caught: unknown) {
      setCredentialError(caught instanceof Error ? caught.message : t("documentBuilder.credentials.errors.clearFailed"));
    } finally {
      setIsUpdatingCredential(false);
    }
  }

  async function extractPreview() {
    if (!file) return;
    setIsExtracting(true);
    setError("");
    setCommitResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("language", language);
      form.append("extraction_method", extractionMethod);
      form.append("min_confidence", String(minConfidence));
      form.append("provider", provider);
      form.append("model", model);
      form.append("ontology_profile", ontologyProfile);
      if (ontologyProfile === "custom") {
        form.append("custom_entity_types", customEntityTypes);
        form.append("custom_relation_types", customRelationTypes);
      }
      const response = await fetch("/api/documents/preview", { method: "POST", body: form });
      const payload: unknown = await response.json();
      if (!response.ok) {
        if (response.status === 503 && extractionMethod === "llm") {
          throw new Error(t("documentBuilder.errors.providerUnavailable", { provider }));
        }
        throw new Error(extractionResponseError(payload, t("documentBuilder.errors.extractFailed")));
      }
      const nextPreview = payload as DocumentPreview;
      setPreview(nextPreview);
      setSelectedEntityIds(new Set(nextPreview.entities.map((entity) => entity.id)));
      setSelectedRelationIds(new Set(nextPreview.relations.map((relation) => relation.id)));
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : t("documentBuilder.errors.extractFailed"));
    } finally {
      setIsExtracting(false);
    }
  }

  async function commitGraph() {
    if (!preview || selectedEntityIds.size === 0) return;
    setIsCommitting(true);
    setError("");
    try {
      const selectedEntities = preview.entities.filter((entity) => selectedEntityIds.has(entity.id));
      const selectedRelations = preview.relations.filter((relation) => effectiveRelationIds.has(relation.id));
      const response = await fetch("/api/documents/commit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: preview.document_id,
          filename: preview.filename,
          media_type: preview.media_type,
          parser: preview.parser,
          language: preview.language,
          extraction_method: preview.extraction_method,
          ontology_profile: preview.ontology_profile,
          character_count: preview.character_count,
          entities: selectedEntities,
          relations: selectedRelations,
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(responseError(payload, t("documentBuilder.errors.commitFailed")));
      }
      setCommitResult(payload as CommitResult);
      await onCommitted?.();
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : t("documentBuilder.errors.commitFailed"));
    } finally {
      setIsCommitting(false);
    }
  }

  function toggleEntity(entityId: string) {
    setSelectedEntityIds((current) => {
      const next = new Set(current);
      if (next.has(entityId)) next.delete(entityId);
      else next.add(entityId);
      return next;
    });
  }

  function toggleRelation(relationId: string) {
    setSelectedRelationIds((current) => {
      const next = new Set(current);
      if (next.has(relationId)) next.delete(relationId);
      else next.add(relationId);
      return next;
    });
  }

  function updateEntityLabel(entityId: string, label: string) {
    setPreview((current) => current ? {
      ...current,
      entities: current.entities.map((entity) => entity.id === entityId ? { ...entity, label } : entity),
    } : current);
  }

  function updateRelationPredicate(relationId: string, predicate: string) {
    setPreview((current) => current ? {
      ...current,
      relations: current.relations.map((relation) => relation.id === relationId ? { ...relation, predicate } : relation),
    } : current);
  }

  return (
    <section className="ws-card" style={{ padding: 24, display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 18, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 42, height: 42, borderRadius: 13, display: "grid", placeItems: "center", color: "#7dd3fc", background: "rgba(14, 116, 144, 0.16)", border: "1px solid rgba(56, 189, 248, 0.26)" }}>
            <FileSearch size={20} />
          </div>
          <div>
            <h2 className="ws-title" style={{ fontSize: 18 }}>{t("documentBuilder.title")}</h2>
            <div className="ws-body" style={{ marginTop: 3 }}>{t("documentBuilder.subtitle")}</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }} aria-label={t("documentBuilder.progressLabel")}>
          {[1, 2, 3].map((step) => (
            <div key={step} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              {step > 1 && <ArrowRight size={12} color="var(--ws-text-dim)" />}
              <div style={{
                width: 26,
                height: 26,
                borderRadius: "50%",
                display: "grid",
                placeItems: "center",
                fontSize: 11,
                fontWeight: 800,
                color: step <= currentStep ? "#e0f2fe" : "var(--ws-text-dim)",
                background: step <= currentStep ? "rgba(14, 165, 233, 0.22)" : "rgba(255,255,255,0.04)",
                border: `1px solid ${step <= currentStep ? "rgba(56,189,248,0.45)" : "var(--ws-border)"}`,
              }}>
                {step < currentStep ? <Check size={13} /> : step}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: preview ? "minmax(280px, 0.72fr) minmax(0, 1.28fr)" : "1fr", gap: 18 }}>
        <div style={{ ...panelStyle, padding: 18, display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <div className="ws-label">{t("documentBuilder.stepUpload")}</div>
            <div
              {...getRootProps()}
              style={{
                marginTop: 8,
                minHeight: 140,
                borderRadius: 12,
                border: `2px dashed ${isDragActive ? "#38bdf8" : "var(--ws-border)"}`,
                background: isDragActive ? "rgba(14,165,233,0.1)" : "rgba(0,0,0,0.18)",
                display: "grid",
                placeItems: "center",
                padding: 22,
                textAlign: "center",
                cursor: "pointer",
              }}
            >
              <input {...getInputProps()} />
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
                {file ? <FileText size={34} color="#4ade80" /> : <UploadCloud size={34} color="#38bdf8" />}
                <div style={{ color: "var(--ws-text)", fontWeight: 700, overflowWrap: "anywhere" }}>
                  {file?.name ?? t("documentBuilder.dropTitle")}
                </div>
                <div className="ws-body" style={{ fontSize: 11 }}>
                  {file ? `${(file.size / 1024).toFixed(1)} KB · ${t("documentBuilder.replaceFile")}` : t("documentBuilder.dropHelp")}
                </div>
              </div>
            </div>
            {fileRejections.length > 0 && (
              <div style={{ color: "#fca5a5", fontSize: 11, marginTop: 7 }}>{t("documentBuilder.errors.invalidFile")}</div>
            )}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <label>
              <span className="ws-label">{t("documentBuilder.language")}</span>
              <select style={fieldStyle} value={language} onChange={(event) => { setLanguage(event.target.value as DocumentLanguage); resetResult(); }}>
                <option value="auto">{t("documentBuilder.languages.auto")}</option>
                <option value="zh">{t("documentBuilder.languages.zh")}</option>
                <option value="en">{t("documentBuilder.languages.en")}</option>
              </select>
            </label>
            <label>
              <span className="ws-label">{t("documentBuilder.method")}</span>
              <select style={fieldStyle} value={extractionMethod} onChange={(event) => { setExtractionMethod(event.target.value as ExtractionMethod); resetResult(); }}>
                <option value="auto">{t("documentBuilder.methods.auto")}</option>
                <option value="rules">{t("documentBuilder.methods.rules")}</option>
                <option value="llm">{t("documentBuilder.methods.llm")}</option>
              </select>
            </label>
          </div>

          <div style={{ color: "var(--ws-text-dim)", fontSize: 10, lineHeight: 1.55 }}>
            {t(`documentBuilder.methodHelp.${extractionMethod}`)}
          </div>

          {extractionMethod !== "rules" && (
            <div style={{ ...panelStyle, padding: 12, display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ display: "grid", gridTemplateColumns: "0.8fr 1.2fr", gap: 10 }}>
                <label>
                  <span className="ws-label">{t("documentBuilder.provider")}</span>
                  <select
                    style={fieldStyle}
                    value={provider}
                    onChange={(event) => {
                      const nextProvider = event.target.value;
                      setProvider(nextProvider);
                      setApiKey("");
                      setCredentialError("");
                      setCredentialNotice("");
                      const capability = providerOptions.find((item) => item.id === nextProvider);
                      setModel(capability?.default_model ?? "");
                      resetResult();
                    }}
                  >
                    {(providerOptions.length ? providerOptions : [{ id: "openai", available: false, default_model: "gpt-4o-mini", reason: "" }]).map((item) => (
                      <option key={item.id} value={item.id}>{item.id}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span className="ws-label">{t("documentBuilder.model")}</span>
                  <input
                    style={fieldStyle}
                    value={model}
                    onChange={(event) => { setModel(event.target.value); resetResult(); }}
                    placeholder={selectedProvider?.default_model ?? "model"}
                  />
                </label>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 7, color: selectedProvider?.available ? "#86efac" : "#fcd34d", fontSize: 10 }}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: selectedProvider?.available ? "#4ade80" : "#f59e0b" }} />
                {selectedProvider?.available
                  ? t("documentBuilder.providerReady")
                  : t("documentBuilder.providerUnavailable", { reason: selectedProviderReason })}
              </div>
              {provider !== "ollama" && (
                <div style={{ borderTop: "1px solid var(--ws-border)", paddingTop: 10, display: "flex", flexDirection: "column", gap: 8 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
                    <span className="ws-label" style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <KeyRound size={12} />
                      {t("documentBuilder.credentials.title")}
                    </span>
                    {selectedProvider?.session_configured && (
                      <span style={{ color: "#86efac", fontSize: 10 }}>{t("documentBuilder.credentials.sessionActive")}</span>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 8, alignItems: "stretch", flexWrap: "wrap" }}>
                    <input
                      type="password"
                      autoComplete="off"
                      autoCapitalize="none"
                      spellCheck={false}
                      aria-label={t("documentBuilder.credentials.apiKey")}
                      style={{ ...fieldStyle, flex: "1 1 220px" }}
                      value={apiKey}
                      onChange={(event) => {
                        setApiKey(event.target.value);
                        setCredentialError("");
                        setCredentialNotice("");
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") void configureSessionCredential();
                      }}
                      placeholder={t("documentBuilder.credentials.placeholder", { provider })}
                    />
                    <button
                      className="ws-btn ws-btn--primary"
                      type="button"
                      disabled={!apiKey.trim() || isUpdatingCredential}
                      onClick={() => void configureSessionCredential()}
                    >
                      {isUpdatingCredential ? <Loader2 size={13} className="ws-spin" /> : <KeyRound size={13} />}
                      {t("documentBuilder.credentials.configure")}
                    </button>
                    {selectedProvider?.session_configured && (
                      <button
                        className="ws-btn"
                        type="button"
                        disabled={isUpdatingCredential}
                        onClick={() => void clearSessionCredential()}
                      >
                        <Trash2 size={13} />
                        {t("documentBuilder.credentials.clear")}
                      </button>
                    )}
                  </div>
                  <div style={{ color: "var(--ws-text-dim)", fontSize: 10, lineHeight: 1.5 }}>
                    {t("documentBuilder.credentials.sessionHelp")}
                  </div>
                  {credentialNotice && <div style={{ color: "#86efac", fontSize: 10 }}>{credentialNotice}</div>}
                  {credentialError && <div style={{ color: "#fca5a5", fontSize: 10 }}>{credentialError}</div>}
                </div>
              )}
            </div>
          )}

          <label>
            <span className="ws-label">{t("documentBuilder.ontologyProfile")}</span>
            <select style={fieldStyle} value={ontologyProfile} onChange={(event) => { setOntologyProfile(event.target.value as OntologyProfile); resetResult(); }}>
              <option value="general">{t("documentBuilder.ontologies.general")}</option>
              <option value="business">{t("documentBuilder.ontologies.business")}</option>
              <option value="biomedical">{t("documentBuilder.ontologies.biomedical")}</option>
              <option value="legal">{t("documentBuilder.ontologies.legal")}</option>
              <option value="custom">{t("documentBuilder.ontologies.custom")}</option>
            </select>
          </label>

          {ontologyProfile === "custom" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <label>
                <span className="ws-label">{t("documentBuilder.customEntityTypes")}</span>
                <textarea style={{ ...fieldStyle, minHeight: 70, resize: "vertical" }} value={customEntityTypes} onChange={(event) => { setCustomEntityTypes(event.target.value); resetResult(); }} />
              </label>
              <label>
                <span className="ws-label">{t("documentBuilder.customRelationTypes")}</span>
                <textarea style={{ ...fieldStyle, minHeight: 70, resize: "vertical" }} value={customRelationTypes} onChange={(event) => { setCustomRelationTypes(event.target.value); resetResult(); }} />
              </label>
            </div>
          )}

          <label>
            <span className="ws-label" style={{ display: "flex", justifyContent: "space-between" }}>
              <span>{t("documentBuilder.confidence")}</span>
              <span>{Math.round(minConfidence * 100)}%</span>
            </span>
            <input
              type="range"
              min="0.2"
              max="0.9"
              step="0.05"
              value={minConfidence}
              onChange={(event) => { setMinConfidence(Number(event.target.value)); resetResult(); }}
              style={{ width: "100%", accentColor: "#38bdf8" }}
            />
          </label>

          <button className="ws-btn ws-btn--primary" onClick={() => void extractPreview()} disabled={!file || isExtracting} style={{ justifyContent: "center" }}>
            {isExtracting ? <><Loader2 size={15} className="ws-spin" />{t("documentBuilder.extracting")}</> : <><Sparkles size={15} />{t("documentBuilder.extract")}</>}
          </button>
        </div>

        {preview && (
          <div style={{ ...panelStyle, padding: 18, minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
              <div>
                <div className="ws-label">{t("documentBuilder.stepReview")}</div>
                <div style={{ color: "var(--ws-text)", fontWeight: 750, marginTop: 5 }}>{preview.filename}</div>
                <div className="ws-body" style={{ fontSize: 11, marginTop: 3 }}>
                  {t("documentBuilder.parsedMeta", {
                    parser: preview.parser,
                    language: preview.language.toUpperCase(),
                    characters: preview.character_count.toLocaleString(locale),
                    chunks: preview.chunk_count,
                  })}
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 7 }}>
                  {preview.execution.actual_methods.map((method) => (
                    <span key={method} style={{ borderRadius: 999, padding: "3px 7px", background: "rgba(56,189,248,0.1)", border: "1px solid rgba(56,189,248,0.24)", color: "#7dd3fc", fontSize: 9 }}>
                      {t("documentBuilder.actualMethod")}: {method}
                    </span>
                  ))}
                  <span style={{ borderRadius: 999, padding: "3px 7px", background: "rgba(167,139,250,0.1)", border: "1px solid rgba(167,139,250,0.24)", color: "#c4b5fd", fontSize: 9 }}>
                    {t("documentBuilder.ontologyBadge")}: {preview.ontology_profile}
                  </span>
                  {preview.execution.fallback_used && (
                    <span style={{ borderRadius: 999, padding: "3px 7px", background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.25)", color: "#fcd34d", fontSize: 9 }}>
                      {t("documentBuilder.fallbackUsed")}
                    </span>
                  )}
                </div>
                <div style={{ color: "var(--ws-text-dim)", fontSize: 9, marginTop: 6 }}>
                  {t("documentBuilder.confidenceHelp")}
                </div>
              </div>
              <button className="ws-btn ws-btn--ghost" onClick={resetResult}><RotateCcw size={13} />{t("documentBuilder.startOver")}</button>
            </div>

            {preview.warnings.length > 0 && (
              <div style={{ borderRadius: 10, padding: "10px 12px", background: "rgba(245,158,11,0.09)", border: "1px solid rgba(245,158,11,0.25)", color: "#fcd34d", fontSize: 11 }}>
                {preview.warnings.map((warning) => <div key={warning}>• {localizedWarning(warning)}</div>)}
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div style={{ borderRadius: 10, padding: 12, background: "rgba(14,165,233,0.08)", border: "1px solid rgba(56,189,248,0.18)" }}>
                <div style={{ fontSize: 22, color: "#7dd3fc", fontWeight: 800 }}>{selectedEntityIds.size}</div>
                <div className="ws-body" style={{ fontSize: 11 }}>{t("documentBuilder.entitiesSelected", { total: preview.entities.length })}</div>
              </div>
              <div style={{ borderRadius: 10, padding: 12, background: "rgba(74,222,128,0.07)", border: "1px solid rgba(74,222,128,0.18)" }}>
                <div style={{ fontSize: 22, color: "#86efac", fontWeight: 800 }}>{effectiveRelationIds.size}</div>
                <div className="ws-body" style={{ fontSize: 11 }}>{t("documentBuilder.relationsSelected", { total: preview.relations.length })}</div>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 0.9fr) minmax(0, 1.1fr)", gap: 12, minHeight: 250 }}>
              <CandidatePanel
                title={t("documentBuilder.entities")}
                actions={(
                  <SelectionActions
                    selectLabel={t("documentBuilder.selectAll")}
                    clearLabel={t("documentBuilder.clearAll")}
                    onSelect={() => setSelectedEntityIds(new Set(preview.entities.map((entity) => entity.id)))}
                    onClear={() => setSelectedEntityIds(new Set())}
                  />
                )}
              >
                {preview.entities.length === 0 ? <EmptyCandidates text={t("documentBuilder.noEntities")} /> : preview.entities.map((entity) => (
                  <label key={entity.id} style={{ display: "grid", gridTemplateColumns: "18px minmax(0,1fr) 62px", gap: 8, alignItems: "center", padding: "9px 10px", borderBottom: "1px solid rgba(255,255,255,0.055)", cursor: "pointer" }}>
                    <input type="checkbox" checked={selectedEntityIds.has(entity.id)} onChange={() => toggleEntity(entity.id)} />
                    <span style={{ minWidth: 0 }}>
                      <span style={{ display: "block", color: "var(--ws-text)", fontSize: 12, fontWeight: 650, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{entity.text}</span>
                      <span style={{ display: "block", color: "var(--ws-text-dim)", fontSize: 10 }}>
                        {t("documentBuilder.mentions", { count: entity.mention_count })} · {Math.round(entity.confidence * 100)}%
                        {Array.isArray(entity.properties.extraction_methods) ? ` · ${entity.properties.extraction_methods.join(", ")}` : ""}
                      </span>
                    </span>
                    <input aria-label={t("documentBuilder.entityTypeFor", { entity: entity.text })} style={{ ...fieldStyle, minHeight: 28, padding: "4px 6px", fontSize: 10 }} value={entity.label} onChange={(event) => updateEntityLabel(entity.id, event.target.value)} />
                  </label>
                ))}
              </CandidatePanel>

              <CandidatePanel
                title={t("documentBuilder.relations")}
                actions={(
                  <SelectionActions
                    selectLabel={t("documentBuilder.selectAll")}
                    clearLabel={t("documentBuilder.clearAll")}
                    onSelect={() => setSelectedRelationIds(new Set(preview.relations.map((relation) => relation.id)))}
                    onClear={() => setSelectedRelationIds(new Set())}
                  />
                )}
              >
                {preview.relations.length === 0 ? <EmptyCandidates text={t("documentBuilder.noRelations")} /> : preview.relations.map((relation) => {
                  const endpointsAvailable = selectedEntityIds.has(relation.source_id) && selectedEntityIds.has(relation.target_id);
                  return (
                    <label key={relation.id} style={{ display: "grid", gridTemplateColumns: "18px minmax(0,1fr)", gap: 8, alignItems: "start", padding: "9px 10px", borderBottom: "1px solid rgba(255,255,255,0.055)", cursor: endpointsAvailable ? "pointer" : "not-allowed", opacity: endpointsAvailable ? 1 : 0.45 }}>
                      <input type="checkbox" disabled={!endpointsAvailable} checked={endpointsAvailable && selectedRelationIds.has(relation.id)} onChange={() => toggleRelation(relation.id)} />
                      <span style={{ minWidth: 0 }}>
                        <span style={{ display: "flex", alignItems: "center", gap: 5, color: "var(--ws-text)", fontSize: 11 }}>
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{entityLabels.get(relation.source_id) ?? relation.source_id}</span>
                          <ArrowRight size={10} />
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{entityLabels.get(relation.target_id) ?? relation.target_id}</span>
                        </span>
                        <span style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) auto", gap: 7, alignItems: "center", marginTop: 5 }}>
                          <input aria-label={t("documentBuilder.relationType")} style={{ ...fieldStyle, minHeight: 27, padding: "4px 6px", fontSize: 10 }} value={relation.predicate} onChange={(event) => updateRelationPredicate(relation.id, event.target.value)} />
                          <span style={{ color: "var(--ws-text-dim)", fontSize: 10 }}>{Math.round(relation.confidence * 100)}%</span>
                        </span>
                        {relation.context && (
                          <span title={relation.context} style={{ display: "block", marginTop: 5, color: "var(--ws-text-dim)", fontSize: 9, lineHeight: 1.45, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {t("documentBuilder.evidence")}: {relation.context}
                          </span>
                        )}
                      </span>
                    </label>
                  );
                })}
              </CandidatePanel>
            </div>

            <details style={{ borderTop: "1px solid var(--ws-border)", paddingTop: 10 }}>
              <summary style={{ color: "var(--ws-text-muted)", fontSize: 11, cursor: "pointer" }}>{t("documentBuilder.textPreview")}</summary>
              <pre style={{ margin: "10px 0 0", maxHeight: 170, overflow: "auto", whiteSpace: "pre-wrap", color: "var(--ws-text-muted)", fontFamily: "inherit", fontSize: 11, lineHeight: 1.6 }}>{preview.text_preview}</pre>
            </details>

            <button className="ws-btn ws-btn--primary" onClick={() => void commitGraph()} disabled={isCommitting || selectedEntityIds.size === 0} style={{ justifyContent: "center" }}>
              {isCommitting ? <><Loader2 size={15} className="ws-spin" />{t("documentBuilder.committing")}</> : <><Network size={15} />{t("documentBuilder.commit", { entities: selectedEntityIds.size, relations: effectiveRelationIds.size })}</>}
            </button>
          </div>
        )}
      </div>

      {error && (
        <div role="alert" style={{ display: "flex", gap: 9, alignItems: "flex-start", borderRadius: 10, padding: "11px 13px", background: "rgba(127,29,29,0.2)", border: "1px solid rgba(248,113,113,0.3)", color: "#fecaca", fontSize: 12 }}>
          <AlertCircle size={15} style={{ flexShrink: 0, marginTop: 1 }} />{error}
        </div>
      )}

      {commitResult && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 14, borderRadius: 12, padding: "14px 16px", background: "rgba(22,101,52,0.18)", border: "1px solid rgba(74,222,128,0.3)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <CheckCircle2 size={19} color="#4ade80" />
            <div>
              <div style={{ color: "#dcfce7", fontWeight: 750, fontSize: 13 }}>{t("documentBuilder.successTitle")}</div>
              <div style={{ color: "#a7f3d0", fontSize: 11, marginTop: 2 }}>{t("documentBuilder.successDetail", { nodes: commitResult.nodes_added, edges: commitResult.edges_added })}</div>
            </div>
          </div>
          {onOpenGraph && <button className="ws-btn" onClick={onOpenGraph}><Network size={14} />{t("documentBuilder.openGraph")}</button>}
        </div>
      )}
    </section>
  );
}

function CandidatePanel({ title, actions, children }: { title: string; actions: ReactNode; children: ReactNode }) {
  return (
    <div style={{ border: "1px solid var(--ws-border)", borderRadius: 10, overflow: "hidden", minWidth: 0, display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, padding: "8px 10px", background: "rgba(255,255,255,0.035)", borderBottom: "1px solid var(--ws-border)" }}>
        <span style={{ color: "var(--ws-text-muted)", fontWeight: 750, fontSize: 11 }}>{title}</span>
        {actions}
      </div>
      <div style={{ maxHeight: 280, overflow: "auto", minHeight: 0 }}>{children}</div>
    </div>
  );
}

function SelectionActions({ selectLabel, clearLabel, onSelect, onClear }: { selectLabel: string; clearLabel: string; onSelect: () => void; onClear: () => void }) {
  return (
    <span style={{ display: "flex", gap: 8 }}>
      <button type="button" onClick={onSelect} style={{ border: 0, padding: 0, background: "transparent", color: "#7dd3fc", fontSize: 10, cursor: "pointer" }}>{selectLabel}</button>
      <button type="button" onClick={onClear} style={{ border: 0, padding: 0, background: "transparent", color: "var(--ws-text-dim)", fontSize: 10, cursor: "pointer" }}>{clearLabel}</button>
    </span>
  );
}

function EmptyCandidates({ text }: { text: string }) {
  return <div style={{ padding: 20, color: "var(--ws-text-dim)", fontSize: 11, textAlign: "center" }}>{text}</div>;
}
