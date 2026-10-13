import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import * as fs from "node:fs/promises";
import * as path from "node:path";

export type ObjectRecord = Record<string, unknown>;
export const RETENTION_MS = 72 * 60 * 60 * 1000;
export const PRUNE_INTERVAL_MS = 5 * 60 * 1000;
const LOCK_STALE_MS = 60 * 60 * 1000;
const MAX_LOCK_RETRIES = 3;
const LOCK_OPEN_FLAGS = constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW;
export function object(value: unknown): ObjectRecord | undefined {
 return value !== null && typeof value === "object" && !Array.isArray(value) ? (value as ObjectRecord) : undefined;
}
export function digest(value: unknown): string {
 return createHash("sha256").update(JSON.stringify(value) ?? "null").digest("hex");
}
export function contentDigest(value: unknown): string {
 return createHash("sha256")
  .update(JSON.stringify(value, (key, val) => (key === "cache_control" ? undefined : val)) ?? "null")
  .digest("hex");
}
export function segment(value: string): string {
 return value.replace(/[^a-zA-Z0-9._-]/g, "_") || "unknown";
}
export function symbols(value: unknown): string[] {
 const record = object(value);
 return record
  ? Object.getOwnPropertySymbols(record)
   .map(s => s.description ?? s.toString())
   .sort()
  : [];
}
export function perCall(value: unknown): boolean {
 const record = object(value);
 return (
  !!record &&
  Object.getOwnPropertySymbols(record).some(
   s => s.description === "agent.message.perCallContext" && Reflect.get(record, s) === true,
  )
 );
}
export interface ItemShape {
 index: number;
 role?: string;
 chars: number;
 hash: string;
 blockCount: number;
 blockHashes: string[];
 cacheMarkers: Array<{ blockIndex: number; control: unknown }>;
 symbolKeys: string[];
 blockSymbolKeys: string[][];
 perCall: boolean;
 clearAt?: unknown;
}
export interface RequestShape {
 system: Array<{
  index: number;
  hash: string;
  chars: number;
  billing: boolean;
  memoryOffset: number;
  cacheControl?: unknown;
 }>;
 systemHash: string;
 toolsHash: string;
 tools: Array<{ index: number; name: unknown; hash: string; cacheControl?: unknown }>;
 items: ItemShape[];
 optionsHash: string;
 options: ObjectRecord;
 lastMarker: number;
 tailMessages: number;
 markers: Array<{ tier: string; index: number; blockIndex?: number; control: unknown }>;
 deltaRequest: boolean;
 bytes: number;
}
export function analyzePayload(payload: unknown): RequestShape {
 const body = object(payload) ?? {};
 let system: unknown[] = [];
 if (Array.isArray(body.system)) system = body.system;
 else if (typeof body.system === "string") system = [{ text: body.system }];
 else if (typeof body.instructions === "string") system = [{ text: body.instructions }];
 const tools: unknown[] = Array.isArray(body.tools) ? body.tools : [];
 let messages: unknown[] = [];
 if (Array.isArray(body.messages)) messages = body.messages;
 else if (Array.isArray(body.input)) messages = body.input;
 const markers: RequestShape["markers"] = [];
 if (body.cache_control !== undefined) markers.push({ tier: "automatic", index: -1, control: body.cache_control });
 const systemShapes = system.map((block, index) => {
  const b = object(block) ?? {};
  const text = typeof b.text === "string" ? b.text : JSON.stringify(block);
  const memory = /(?:^|[\n,])<memories>\r?\n[\s\S]*?<\/memories>/.exec(text);
  if (b.cache_control !== undefined) markers.push({ tier: "system", index, control: b.cache_control });
  return {
   index,
   hash: contentDigest(block),
   chars: text.length,
   billing: text.startsWith("x-anthropic-billing-header:"),
   memoryOffset: memory ? memory.index + memory[0].indexOf("<memories>") : -1,
   cacheControl: b.cache_control,
  };
 });
 const toolShapes = tools.map((tool, index) => {
  const t = object(tool) ?? {};
  if (t.cache_control !== undefined) markers.push({ tier: "tool", index, control: t.cache_control });
  return {
   index,
   name: t.name ?? object(t.function)?.name ?? t.type,
   hash: contentDigest(tool),
   cacheControl: t.cache_control,
  };
 });
 const items = messages.map((message, index): ItemShape => {
  const m = object(message) ?? {};
  const blocks: unknown[] = Array.isArray(m.content) ? m.content : [m.content ?? message];
  const cacheMarkers = blocks.flatMap((block, blockIndex) => {
   const control = object(block)?.cache_control;
   if (control === undefined) return [];
   markers.push({ tier: "message", index, blockIndex, control });
   return [{ blockIndex, control }];
  });
  return {
   index,
   role: typeof m.role === "string" ? m.role : undefined,
   chars: JSON.stringify(message).length,
   hash: contentDigest(message),
   blockCount: blocks.length,
   blockHashes: blocks.map(contentDigest),
   cacheMarkers,
   symbolKeys: symbols(message),
   blockSymbolKeys: blocks.map(symbols),
   perCall: perCall(message),
   clearAt: m.clear_at,
  };
 });
 const options = Object.fromEntries(
  Object.entries(body).filter(
   ([key]) => !["system", "instructions", "tools", "messages", "input", "metadata", "diagnostics"].includes(key),
  ),
 );
 const lastMarker = Math.max(-1, ...markers.filter(m => m.tier === "message").map(m => m.index));
 return {
  system: systemShapes,
  systemHash: digest(systemShapes.filter(s => !s.billing).map(s => s.hash)),
  toolsHash: digest(toolShapes.map(t => t.hash)),
  tools: toolShapes,
  items,
  optionsHash: digest(options),
  options,
  lastMarker,
  tailMessages: items.length - 1 - lastMarker,
  markers,
  deltaRequest: typeof body.previous_response_id === "string",
  bytes: Buffer.byteLength(JSON.stringify(payload) ?? "null"),
 };
}
const shapes = new WeakMap<object, RequestShape>();
/** {@link analyzePayload} once per payload object, shared by every handler of the same request hook. */
export function analyzePayloadOnce(payload: unknown): RequestShape {
 const key = object(payload);
 if (!key) return analyzePayload(payload);
 let shape = shapes.get(key);
 if (!shape) {
  shape = analyzePayload(payload);
  shapes.set(key, shape);
 }
 return shape;
}
/** True when every message of `older` sits unchanged at the same position at the start of `newer`. */
export function historyContains(older: readonly ItemShape[], newer: readonly ItemShape[]): boolean {
 if (older.length > newer.length) return false;
 for (let index = 0; index < older.length; index++) {
  if (older[index]!.hash !== newer[index]!.hash) return false;
 }
 return true;
}
export function compareShapes(previous: RequestShape, current: RequestShape): ObjectRecord {
 let matched = 0;
 while (
  matched < Math.min(previous.items.length, current.items.length) &&
  previous.items[matched]?.hash === current.items[matched]?.hash
 )
  matched++;
 const oldItem = previous.items[matched],
  newItem = current.items[matched];
 let matchedBlocks = 0;
 if (oldItem && newItem)
  while (
   matchedBlocks < Math.min(oldItem.blockHashes.length, newItem.blockHashes.length) &&
   oldItem.blockHashes[matchedBlocks] === newItem.blockHashes[matchedBlocks]
  )
   matchedBlocks++;
 return {
  systemChanged: previous.systemHash !== current.systemHash,
  toolsChanged: previous.toolsHash !== current.toolsHash,
  optionsChanged: previous.optionsHash !== current.optionsHash,
  messagePrefixMatched: matched,
  previousItemCount: previous.items.length,
  messagesRewritten: matched < previous.items.length && !current.deltaRequest,
  deltaRequest: current.deltaRequest,
  firstChangedBlock: oldItem && newItem ? matchedBlocks : null,
  systemChangedIndices: current.system
   .filter((s, i) => !s.billing && s.hash !== previous.system[i]?.hash)
   .map(s => s.index),
 };
}

export async function atomicWrite(file: string, text: string): Promise<void> {
 await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
 const temporary = `${file}.${randomUUID()}.tmp`;
 const handle = await fs.open(temporary, "wx", 0o600);
 try {
  try {
   await handle.writeFile(text);
  } finally {
   await handle.close();
  }
  await fs.rename(temporary, file);
 } finally {
  await fs.rm(temporary, { force: true });
 }
}

export interface PruneResult {
 cutoff: string;
 deleted: number;
 skipped: number;
 errors: string[];
 locked: boolean;
}
/** `HHMMSS.mmm-…-<kind>.json`, or its interrupted `atomicWrite` temporary. The day directory holds the UTC date. */
const KNOWN_JSON =
 /^(\d{2})(\d{2})(\d{2})\.(\d{3})-.+-(?:request|wire_request|response|message|compaction|snapshot|lifecycle|gap|retention|summary)\.json(?:\.[\da-f-]{36}\.tmp)?$/;
/** `HHMM-…-stream.jsonl`: every line in it was written within that UTC minute. */
const STREAM = /^(\d{2})(\d{2})-.+-stream\.jsonl$/;

/** Latest write time a capture file can hold, from its name alone; `undefined` for files the capture does not own. */
function captureEnd(day: string, name: string): number | undefined {
 const json = KNOWN_JSON.exec(name);
 if (json) return Date.parse(`${day}T${json[1]}:${json[2]}:${json[3]}.${json[4]}Z`);
 const stream = STREAM.exec(name);
 if (stream) return Date.parse(`${day}T${stream[1]}:${stream[2]}:00.000Z`) + 59_999;
 return undefined;
}

/**
 * A claim inside the directory makes it non-empty, excluding other claimants
 * and legacy rmdir-based release. Recheck the inode after claiming: the path
 * may have been released/recreated before mkdir reached it.
 * The caller holds the directory open so its inode cannot be recycled during this operation.
 */
async function removeOwnedLock(lock: string, owner: { dev: number; ino: number }): Promise<boolean> {
 const claim = path.join(lock, ".owner-operation");
 try {
  await fs.mkdir(claim, { mode: 0o700 });
 } catch (error) {
  if (["ENOENT", "EEXIST"].includes(String(object(error)?.code))) return false;
  throw error;
 }
 let moved = false;
 try {
  const current = await fs.lstat(lock);
  if (current.dev !== owner.dev || current.ino !== owner.ino) return false;
  const retired = `${lock}.${randomUUID()}`;
  await fs.rename(lock, retired);
  moved = true;
  await fs.rm(retired, { recursive: true, force: true });
  return true;
 } finally {
  if (!moved) await fs.rmdir(claim).catch(error => {
   if (object(error)?.code !== "ENOENT") throw error;
  });
 }
}

/**
 * Deletes captures past retention using only directory listings and file names; never reads a capture.
 * Only recognized capture files beneath real provider/date directories are owned. Never follow symlinks.
 */
export async function pruneCaptures(root: string, now = Date.now(), dryRun = false): Promise<PruneResult> {
 const cutoff = now - RETENTION_MS;
 const cutoffDay = new Date(cutoff).toISOString().slice(0, 10);
 const result: PruneResult = {
  cutoff: new Date(cutoff).toISOString(),
  deleted: 0,
  skipped: 0,
  errors: [],
  locked: false,
 };
 if (!dryRun) await fs.mkdir(root, { recursive: true, mode: 0o700 });
 try {
  const stat = await fs.lstat(root);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error("Capture root must be a real directory");
 } catch (error) {
  if (dryRun && object(error)?.code === "ENOENT") return result;
  throw error;
 }
 const lock = path.join(root, ".retention-lock");
 let owner: { dev: number; ino: number } | undefined;
 let ownerHandle: fs.FileHandle | undefined;
 if (!dryRun) {
  for (let attempt = 0; attempt < MAX_LOCK_RETRIES && !owner; attempt++) {
   try {
    await fs.mkdir(lock, { mode: 0o700 });
    ownerHandle = await fs.open(lock, LOCK_OPEN_FLAGS);
    owner = await ownerHandle.stat();
   } catch (error) {
    await ownerHandle?.close();
    ownerHandle = undefined;
    if (object(error)?.code === "ENOENT") continue;
    if (object(error)?.code !== "EEXIST") throw error;
    try {
     const staleHandle = await fs.open(lock, LOCK_OPEN_FLAGS);
     try {
      const previous = await staleHandle.stat();
      if (!previous.isDirectory() || Date.now() - previous.mtimeMs < LOCK_STALE_MS) break;
      if (!await removeOwnedLock(lock, previous)) break;
     } finally {
      await staleHandle.close();
     }
    } catch (error) {
     if (object(error)?.code === "ENOENT") continue;
     result.errors.push(`${lock}: ${String(error)}`);
     break;
    }
   }
  }
  if (!owner) {
   result.locked = true;
   return result;
  }
 }
 try {
  for (const provider of await fs.readdir(root, { withFileTypes: true })) {
   if (!provider.isDirectory() || provider.name.startsWith(".")) continue;
   const providerDir = path.join(root, provider.name);
   try {
    for (const day of await fs.readdir(providerDir, { withFileTypes: true })) {
     if (!day.isDirectory() || !/^\d{4}-\d{2}-\d{2}$/.test(day.name) || day.name > cutoffDay) continue;
     const dir = path.join(providerDir, day.name);
     try {
      for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
       const end = entry.isFile() ? captureEnd(day.name, entry.name) : undefined;
       if (end === undefined || !Number.isFinite(end)) {
        result.skipped++;
        continue;
       }
       if (end >= cutoff) continue;
       const file = path.join(dir, entry.name);
       try {
        if (!dryRun) await fs.unlink(file);
        result.deleted++;
       } catch (error) {
        if (object(error)?.code !== "ENOENT") result.errors.push(`${file}: ${String(error)}`);
       }
      }
      if (!dryRun) {
       try {
        await fs.rmdir(dir);
       } catch (error) {
        if (!["ENOENT", "ENOTEMPTY", "EEXIST"].includes(String(object(error)?.code))) result.errors.push(`${dir}: ${String(error)}`);
       }
      }
     } catch (error) {
      result.errors.push(`${dir}: ${String(error)}`);
     }
    }
   } catch (error) {
    result.errors.push(`${providerDir}: ${String(error)}`);
   }
  }
 } finally {
  if (owner) await removeOwnedLock(lock, owner).catch(error => { result.errors.push(`${lock}: ${String(error)}`); });
  await ownerHandle?.close().catch(error => { result.errors.push(`${lock}: ${String(error)}`); });
 }
 return result;
}
