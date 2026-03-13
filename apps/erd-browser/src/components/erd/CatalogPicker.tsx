import { useState } from "react";
import { Database, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { setCatalogInUrl } from "@/catalog-config";

const EXAMPLE_CATALOGS = [
  {
    label: "FaceBase (production)",
    hostname: "www.facebase.org",
    catalogId: "1",
    description: "Craniofacial research data — public access",
  },
  {
    label: "FaceBase (dev)",
    hostname: "dev.facebase.org",
    catalogId: "1",
    description: "FaceBase development server — public access",
  },
  {
    label: "MusMorph ML (dev)",
    hostname: "dev.facebase.org",
    catalogId: "10",
    description: "Mouse morphometry ML catalog — requires authentication",
  },
];

export default function CatalogPicker() {
  const [hostname, setHostname] = useState("");
  const [catalogId, setCatalogId] = useState("");

  const handleConnect = () => {
    if (hostname && catalogId) {
      setCatalogInUrl(hostname.trim(), catalogId.trim());
    }
  };

  const handleExample = (host: string, catalog: string) => {
    setCatalogInUrl(host, catalog);
  };

  return (
    <div className="h-screen flex items-center justify-center bg-slate-50">
      <div className="max-w-lg w-full px-6">
        <div className="text-center mb-8">
          <Database className="h-10 w-10 mx-auto mb-3 text-slate-400" />
          <h1 className="text-xl font-bold text-slate-900">
            Deriva ERD Browser
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Explore the schema of any Deriva catalog
          </p>
        </div>

        {/* Manual entry */}
        <div className="bg-white border border-slate-200 rounded-lg p-5 mb-4">
          <h2 className="text-sm font-semibold text-slate-700 mb-3">
            Connect to a catalog
          </h2>
          <div className="flex gap-2">
            <Input
              placeholder="Hostname (e.g., www.facebase.org)"
              value={hostname}
              onChange={(e) => setHostname(e.target.value)}
              className="text-sm"
              onKeyDown={(e) => e.key === "Enter" && handleConnect()}
            />
            <Input
              placeholder="Catalog ID"
              value={catalogId}
              onChange={(e) => setCatalogId(e.target.value)}
              className="text-sm w-28 flex-shrink-0"
              onKeyDown={(e) => e.key === "Enter" && handleConnect()}
            />
            <Button
              onClick={handleConnect}
              disabled={!hostname || !catalogId}
              className="flex-shrink-0"
            >
              Connect
            </Button>
          </div>
          <p className="text-xs text-slate-400 mt-2">
            Public catalogs work directly. Private catalogs require CORS
            configuration on the server, or use the{" "}
            <span className="font-mono">Vite dev proxy</span> locally.
          </p>
        </div>

        {/* Example catalogs */}
        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-slate-700 mb-3">
            Public catalogs
          </h2>
          <div className="space-y-2">
            {EXAMPLE_CATALOGS.map((ex) => (
              <button
                key={`${ex.hostname}/${ex.catalogId}`}
                onClick={() => handleExample(ex.hostname, ex.catalogId)}
                className="w-full text-left px-3 py-2.5 rounded-md border border-slate-100 hover:border-slate-300 hover:bg-slate-50 transition-colors group"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-slate-800">
                    {ex.label}
                  </span>
                  <ExternalLink className="h-3.5 w-3.5 text-slate-300 group-hover:text-slate-500" />
                </div>
                <div className="text-xs text-slate-400 mt-0.5">
                  {ex.hostname} / #{ex.catalogId}
                  {ex.description && ` — ${ex.description}`}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
