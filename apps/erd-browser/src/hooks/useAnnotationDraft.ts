/**
 * Draft state management for annotation editing.
 *
 * Instead of writing every change immediately to the catalog,
 * edits accumulate in a local draft. The user explicitly saves
 * (which batches all PUTs/DELETEs) or discards changes.
 *
 * An undo history lets the user restore the previous server state
 * after saving.
 */

import { useState, useCallback, useRef } from "react";

/** A single pending change to an annotation */
export type AnnotationChange =
  | { type: "add"; tag: string; value: any }
  | { type: "update"; tag: string; value: any }
  | { type: "delete"; tag: string };

export interface AnnotationDraft {
  /** The live working copy of annotations (server state + local edits) */
  annotations: Record<string, any>;

  /** Whether the draft differs from the last-saved server state */
  isDirty: boolean;

  /** Number of pending changes */
  changeCount: number;

  /** Add a new annotation tag with an empty value */
  addTag: (tag: string) => void;

  /** Delete an annotation tag */
  deleteTag: (tag: string) => void;

  /** Update the value of an annotation tag */
  updateTag: (tag: string, value: any) => void;

  /** Discard all local edits and revert to server state */
  discard: () => void;

  /** Whether an undo snapshot is available */
  canUndo: boolean;

  /** Restore the previous server state (undo the last save) */
  undoSnapshot: Record<string, any> | null;

  /** Mark the undo as consumed */
  clearUndo: () => void;

  /** Get the list of pending changes for the save operation */
  getChanges: () => AnnotationChange[];

  /** Called after a successful save — promotes draft to server state */
  commitSave: () => void;
}

/**
 * Hook to manage a local draft of an annotation map.
 *
 * @param serverAnnotations — the annotations object as loaded from the catalog.
 *   This is treated as the "source of truth" baseline. The hook deep-clones it
 *   on init and tracks edits on top of the clone.
 */
export function useAnnotationDraft(
  serverAnnotations: Record<string, any>
): AnnotationDraft {
  // Deep-clone the server state as baseline
  const baselineRef = useRef<Record<string, any>>(
    JSON.parse(JSON.stringify(serverAnnotations))
  );

  // Working copy starts as a clone of server state
  const [draft, setDraft] = useState<Record<string, any>>(() =>
    JSON.parse(JSON.stringify(serverAnnotations))
  );

  // Undo: snapshot of annotations before the last save
  const [undoSnapshot, setUndoSnapshot] = useState<Record<string, any> | null>(
    null
  );

  const addTag = useCallback((tag: string) => {
    setDraft((prev) => {
      if (tag in prev) return prev; // already exists
      return { ...prev, [tag]: {} };
    });
  }, []);

  const deleteTag = useCallback((tag: string) => {
    setDraft((prev) => {
      if (!(tag in prev)) return prev;
      const next = { ...prev };
      delete next[tag];
      return next;
    });
  }, []);

  const updateTag = useCallback((tag: string, value: any) => {
    setDraft((prev) => ({ ...prev, [tag]: value }));
  }, []);

  const discard = useCallback(() => {
    setDraft(JSON.parse(JSON.stringify(baselineRef.current)));
  }, []);

  const clearUndo = useCallback(() => {
    setUndoSnapshot(null);
  }, []);

  // Compute changes by diffing draft against baseline
  const getChanges = useCallback((): AnnotationChange[] => {
    const baseline = baselineRef.current;
    const changes: AnnotationChange[] = [];

    // Tags in draft but not in baseline → add
    // Tags in both but value changed → update
    for (const [tag, value] of Object.entries(draft)) {
      if (!(tag in baseline)) {
        changes.push({ type: "add", tag, value });
      } else if (JSON.stringify(value) !== JSON.stringify(baseline[tag])) {
        changes.push({ type: "update", tag, value });
      }
    }

    // Tags in baseline but not in draft → delete
    for (const tag of Object.keys(baseline)) {
      if (!(tag in draft)) {
        changes.push({ type: "delete", tag });
      }
    }

    return changes;
  }, [draft]);

  // After a successful save, promote draft to the new baseline
  const commitSave = useCallback(() => {
    // Snapshot old baseline for undo
    setUndoSnapshot(JSON.parse(JSON.stringify(baselineRef.current)));
    // Promote current draft as the new baseline
    baselineRef.current = JSON.parse(JSON.stringify(draft));
  }, [draft]);

  const isDirty =
    JSON.stringify(draft) !== JSON.stringify(baselineRef.current);

  const changeCount = isDirty ? getChanges().length : 0;

  return {
    annotations: draft,
    isDirty,
    changeCount,
    addTag,
    deleteTag,
    updateTag,
    discard,
    canUndo: undoSnapshot !== null,
    undoSnapshot,
    clearUndo,
    getChanges,
    commitSave,
  };
}
