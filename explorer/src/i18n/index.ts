import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import en from "./resources/en";
import zhCN from "./resources/zh-CN";

export const LANGUAGE_STORAGE_KEY = "semantica.language";
export const supportedLanguages = ["en", "zh-CN"] as const;
export type SupportedLanguage = (typeof supportedLanguages)[number];

function normalizeLanguage(language?: string | null): SupportedLanguage {
  return language?.toLowerCase().startsWith("zh") ? "zh-CN" : "en";
}

function getInitialLanguage(): SupportedLanguage {
  if (typeof window === "undefined") {
    return "en";
  }

  return normalizeLanguage(
    window.localStorage.getItem(LANGUAGE_STORAGE_KEY) ?? window.navigator.language,
  );
}

void i18n.use(initReactI18next).init({
  resources: { en, "zh-CN": zhCN },
  lng: getInitialLanguage(),
  fallbackLng: "en",
  supportedLngs: supportedLanguages,
  interpolation: { escapeValue: false },
});

i18n.on("languageChanged", (language) => {
  const normalized = normalizeLanguage(language);

  if (typeof document !== "undefined") {
    document.documentElement.lang = normalized;
  }

  if (typeof window !== "undefined") {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, normalized);
  }
});

if (typeof document !== "undefined") {
  document.documentElement.lang = normalizeLanguage(i18n.resolvedLanguage);
}

export function getSupportedLanguage(language?: string | null): SupportedLanguage {
  return normalizeLanguage(language);
}

export default i18n;
