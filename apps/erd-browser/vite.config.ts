import path from "path";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig } from "vite";

const catalogHost = process.env.VITE_CATALOG_HOST || "localhost";
const protocol = catalogHost === "localhost" ? "https" : "https";
const target = `${protocol}://${catalogHost}`;

export default defineConfig({
  plugins: [react(), basicSsl()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    open: true,
    proxy: {
      "/ermrest": {
        target,
        changeOrigin: true,
        secure: false, // Allow self-signed certs on localhost
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
