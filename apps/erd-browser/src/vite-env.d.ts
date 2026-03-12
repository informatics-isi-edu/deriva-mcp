/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CATALOG_HOST: string;
  readonly VITE_CATALOG_ID: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
