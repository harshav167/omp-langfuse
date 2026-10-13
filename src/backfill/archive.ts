import { constants, createWriteStream } from "node:fs";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { tmpdir } from "node:os";
import { pipeline } from "node:stream/promises";
import { StringDecoder } from "node:string_decoder";
import type { BackfillReport } from "../backfill.ts";

export interface CaptureSnapshot {
  root: string;
  files: CaptureFile[];
}

export interface SnapshotSelection {
  sessions?: ReadonlySet<string>;
  limit?: number;
  onSkip(): void;
}

/** Preserve atomically written records against the active logger's filename-based pruning. */
export async function snapshotCaptures(
  files: readonly CaptureFile[], onMissing: () => void, selection: SnapshotSelection,
): Promise<CaptureSnapshot> {
  const root = await fs.mkdtemp(path.join(tmpdir(), "omp-backfill-snapshot-"));
  // The private directory protects hard-linked files without changing their source inode.
  const directories = new Set<string>();
  const preserved: CaptureFile[] = [];
  let requests = 0;
  try {
    for (const source of files) {
      if (selection.sessions) {
        try {
          const sessionId = await peekSessionId(source);
          if (sessionId && !selection.sessions.has(sessionId)) { selection.onSkip(); continue; }
        } catch (error) {
          const code = (error as NodeJS.ErrnoException).code;
          if (code === "ENOENT" || code === "ELOOP") { onMissing(); continue; }
          throw error;
        }
      }
      if (source.kind === "request") {
        if (selection.limit !== undefined && requests >= selection.limit) break;
        requests++;
      }
      const target = path.join(root, source.relativeFile);
      const directory = path.dirname(target);
      if (!directories.has(directory)) {
        await fs.mkdir(directory, { recursive: true, mode: 0o700 });
        directories.add(directory);
      }
      try {
        try {
          await fs.link(source.file, target);
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "EXDEV") throw error;
          // Copy from the O_NOFOLLOW-checked handle, not a path that can be
          // replaced by a symlink between enumeration and copyFile.
          const handle = await openOwnedFile(source);
          try {
            await pipeline(handle.createReadStream({ autoClose: false }), createWriteStream(target, { flags: "wx", mode: 0o600 }));
          } finally {
            await handle.close();
          }
        }
        preserved.push({ ...source, file: target });
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code === "ENOENT" || code === "ELOOP") { onMissing(); continue; }
        throw new Error(`Could not snapshot ${source.relativeFile}`, { cause: error });
      }
    }
    return { root, files: preserved };
  } catch (error) {
    try {
      await fs.rm(root, { recursive: true, force: true });
    } catch (cleanupError) {
      throw new AggregateError([error, cleanupError], `Snapshot construction failed and cleanup failed at ${root}`);
    }
    throw error;
  }
}
const CAPTURE_FILE = /^(\d{6})\.(\d{3})-.+-(\d+)-([a-z][a-z\d_]*)\.json$/;
const SESSION_PREFIX_LIMIT = 64 * 1024;
export interface CaptureFile {
  file: string;
  relativeFile: string;
  kind: string;
  sequence: number;
  filenameTime: number;
  stat: { birthtimeMs: number; ctimeMs: number; mtimeMs: number };
}
function parseCaptureFile(root: string, provider: string, day: string, name: string): CaptureFile | undefined {
  const match = CAPTURE_FILE.exec(name);
  if (!match) return undefined;
  const dayMs = Date.parse(`${day}T00:00:00.000Z`);
  if (!Number.isFinite(dayMs) || new Date(dayMs).toISOString().slice(0, 10) !== day) return undefined;
  const timeText = `${match[1]!.slice(0, 2)}:${match[1]!.slice(2, 4)}:${match[1]!.slice(4, 6)}.${match[2]}Z`;
  const filenameTime = Date.parse(`${day}T${timeText}`);
  const sequence = Number(match[3]);
  if (!Number.isFinite(filenameTime) || !Number.isSafeInteger(sequence)) return undefined;
  const file = path.join(root, provider, day, name);
  return {
    file,
    relativeFile: path.join(provider, day, name),
    kind: match[4]!,
    sequence,
    filenameTime,
    stat: { birthtimeMs: 0, ctimeMs: 0, mtimeMs: 0 },
  };
}
export async function enumerateCaptureFiles(
  root: string,
  startupTime: number,
  from: number | undefined,
  until: number,
  report: Pick<BackfillReport, "skipped" | "incomplete">,
): Promise<CaptureFile[]> {
  let rootStat;
  try {
    rootStat = await fs.lstat(root);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
  if (rootStat.isSymbolicLink() || !rootStat.isDirectory()) {
    throw new Error("Raw capture root must be a real directory, not a symlink");
  }

  const files: CaptureFile[] = [];
  for (const provider of await fs.readdir(root, { withFileTypes: true })) {
    if (!provider.isDirectory() || provider.name.startsWith(".")) continue;
    const providerPath = path.join(root, provider.name);
    for (const day of await fs.readdir(providerPath, { withFileTypes: true })) {
      if (!day.isDirectory() || !/^\d{4}-\d\d-\d\d$/.test(day.name)) continue;
      const dayPath = path.join(providerPath, day.name);
      for (const entry of await fs.readdir(dayPath, { withFileTypes: true })) {
        if (!entry.isFile()) continue;
        let file = parseCaptureFile(root, provider.name, day.name, entry.name);
        if (!file) continue;
        if (file.filenameTime < (from ?? Number.NEGATIVE_INFINITY) || file.filenameTime > until) {
          report.skipped++;
          continue;
        }
        try {
          const stat = await fs.lstat(file.file);
          if (stat.isSymbolicLink() || !stat.isFile()) continue;
          file = {
            ...file,
            stat: {
              birthtimeMs: stat.birthtimeMs,
              ctimeMs: stat.ctimeMs,
              mtimeMs: stat.mtimeMs,
            },
          };
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code === "ENOENT") {
            report.skipped++;
            report.incomplete++;
            continue;
          }
          throw error;
        }
        // The startup snapshot excludes atomic writes that appear while this walk is in progress,
        // even when their archived event timestamp is earlier than the cutoff.
        const addedAt = Math.max(file.stat.birthtimeMs || 0, file.stat.ctimeMs || 0, file.stat.mtimeMs || 0);
        // Date.now(), archive filenames, and ISO selections have millisecond
        // precision; sub-millisecond filesystem fractions are not newer records.
        if (Math.floor(addedAt) > startupTime) {
          report.skipped++;
          continue;
        }
        files.push(file);
      }
    }
  }
  files.sort((a, b) =>
    a.filenameTime - b.filenameTime || a.sequence - b.sequence || a.relativeFile.localeCompare(b.relativeFile),
  );
  return files;
}
async function openOwnedFile(file: CaptureFile): Promise<fs.FileHandle> {
  const handle = await fs.open(file.file, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const stat = await handle.stat();
    if (!stat.isFile()) throw new Error("Capture changed from a regular file while being opened");
  } catch (error) {
    await handle.close();
    throw error;
  }
  return handle;
}
export async function readCapture(file: CaptureFile): Promise<unknown> {
  const handle = await openOwnedFile(file);
  try {
    return JSON.parse(await handle.readFile("utf8")) as unknown;
  } finally {
    await handle.close();
  }
}
export async function peekSessionId(file: CaptureFile): Promise<string | undefined> {
  const handle = await openOwnedFile(file);
  try {
    const chunk = Buffer.allocUnsafe(4096);
    const decoder = new StringDecoder("utf8");
    let text = "";
    let position = 0;
    while (position < SESSION_PREFIX_LIMIT) {
      const { bytesRead } = await handle.read(chunk, 0, Math.min(chunk.length, SESSION_PREFIX_LIMIT - position), position);
      if (!bytesRead) return undefined;
      text += decoder.write(chunk.subarray(0, bytesRead));
      position += bytesRead;
      const match = /"sessionId"\s*:\s*"((?:\\.|[^"\\])*)"/.exec(text);
      if (match) {
        try {
          return JSON.parse(`"${match[1]}"`) as string;
        } catch {
          return undefined;
        }
      }
      // The raw-capture writer places identity.sessionId before payload; once the payload key begins,
      // there is no session ID in this prefix if the record omitted identity.
      if (text.includes('"payload"')) return undefined;
    }
    return undefined;
  } finally {
    await handle.close();
  }
}
