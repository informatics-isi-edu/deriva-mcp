import path from "path";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig } from "vite";

const catalogHost = process.env.VITE_CATALOG_HOST || "localhost";
const target = `https://${catalogHost}`;

export default defineConfig({
  // For GitHub Pages: set to repo name (e.g., "/erd-browser/")
  // For local dev or root deployment: leave as "/"
  base: process.env.GITHUB_PAGES ? "/deriva-mcp/" : "/",
  plugins: [react(), basicSsl()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    open: true,
    proxy: {
      // Only used in local dev — proxies requests to the Deriva server
      // to avoid CORS issues with credentialed requests.
      "/ermrest": {
        target,
        changeOrigin: true,
        secure: false,
      },
      "/chaise": {
        target,
        changeOrigin: true,
        secure: false,
      },
      "/authn": {
        target,
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
