import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Python virtualenv -- ESLint has no built-in awareness of it (unlike node_modules),
    // so without this it crawls bundled .js/.mjs assets inside torch/sklearn/coverage
    // and reports their lint issues as if they were project code.
    ".venv/**",
  ]),
]);

export default eslintConfig;
