import assert from "node:assert/strict";
import test from "node:test";

import i18n, { getSupportedLanguage, supportedLanguages } from "../src/i18n/index.ts";
import en from "../src/i18n/resources/en.ts";
import zhCN from "../src/i18n/resources/zh-CN.ts";

function leafKeys(value: unknown, prefix = ""): string[] {
  if (typeof value !== "object" || value === null) {
    return [prefix];
  }

  return Object.entries(value).flatMap(([key, child]) =>
    leafKeys(child, prefix ? `${prefix}.${key}` : key),
  );
}

test("English and Chinese resources have matching keys", () => {
  assert.deepEqual(
    leafKeys(zhCN).sort(),
    leafKeys(en).sort(),
  );
});

test("language normalization maps Chinese variants and defaults to English", () => {
  assert.deepEqual(supportedLanguages, ["en", "zh-CN"]);
  assert.equal(getSupportedLanguage("zh-CN"), "zh-CN");
  assert.equal(getSupportedLanguage("zh-TW"), "zh-CN");
  assert.equal(getSupportedLanguage("en-US"), "en");
  assert.equal(getSupportedLanguage("fr-FR"), "en");
});

test("runtime language switching translates labels and interpolation", async () => {
  await i18n.changeLanguage("zh-CN");
  assert.equal(i18n.t("nav.explore.label"), "知识探索");
  assert.equal(i18n.t("welcome.coverage", { value: "66%" }), "覆盖率 66%");

  await i18n.changeLanguage("en");
  assert.equal(i18n.t("nav.explore.label"), "Knowledge Explorer");
  assert.equal(i18n.t("welcome.coverage", { value: "66%" }), "66% coverage");
});
