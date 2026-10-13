// `npm run bundle`: decision.js and its tokenizer dependency in one ES module (bundle/decision.js), for pages
// that load it without a package manager.
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: { entry: "src/index.ts", formats: ["es"], fileName: () => "decision.js" },
    outDir: "bundle",
    emptyOutDir: true,
    target: "es2022",
  },
});
