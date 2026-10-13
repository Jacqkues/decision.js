// Python's text renderings, as d1's prompt.py writes them into prompts: json.dumps (ensure_ascii=False) and
// str() in f-strings. JavaScript has one number type: integral numbers render as Python ints, others as
// Python floats (repr), so a Python float like 1.0 cannot be told from the int 1.

/** repr() of a Python float (shortest round-trip digits, exponent below 1e-4 and from 1e16). */
export function pyFloat(x: number): string {
  if (Number.isNaN(x)) return "NaN";
  if (!Number.isFinite(x)) return x > 0 ? "Infinity" : "-Infinity";
  if (Number.isInteger(x) && Math.abs(x) < 1e16) return String(x);
  const [mant, expStr] = x.toExponential().split("e") as [string, string];
  const exp = Number(expStr);
  if (exp < -4 || exp >= 16) return `${mant}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  const digits = mant.replace("-", "").replace(".", "");
  const sign = x < 0 ? "-" : "";
  if (exp < 0) return `${sign}0.${"0".repeat(-exp - 1)}${digits}`;
  const int = digits.slice(0, exp + 1).padEnd(exp + 1, "0"), frac = digits.slice(exp + 1);
  return `${sign}${int}.${frac || "0"}`;
}

const isInt = (x: number) => Number.isInteger(x) && Math.abs(x) < 1e16;

/** json.dumps(value, ensure_ascii=False) with Python's default separators, or with `indent`. */
export function pyDumps(v: unknown, indent?: number, level = 0): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return isInt(v) ? String(v) : pyFloat(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (typeof v === "bigint") return v.toString();
  const items: string[] = Array.isArray(v)
    ? v.map((x) => pyDumps(x, indent, level + 1))
    : Object.entries(v as object).map(([k, x]) => `${JSON.stringify(k)}: ${pyDumps(x, indent, level + 1)}`);
  const [open, close] = Array.isArray(v) ? ["[", "]"] : ["{", "}"];
  if (!items.length) return open + close;
  if (indent === undefined) return open + items.join(", ") + close;
  const pad = "\n" + " ".repeat(indent * (level + 1));
  return open + pad + items.join("," + pad) + "\n" + " ".repeat(indent * level) + close;
}

/** repr() of a str: single quotes unless the text holds one and no double quote. */
function pyReprStr(s: string) {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = q;
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    if (ch === "\\") out += "\\\\";
    else if (ch === q) out += "\\" + q;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (c < 0x20 || c === 0x7f) out += "\\x" + c.toString(16).padStart(2, "0");
    else out += ch;
  }
  return out + q;
}

function pyRepr(v: unknown): string {
  return typeof v === "string" ? pyReprStr(v) : pyStr(v);
}

/** str() of a JSON value, as an f-string writes it. */
export function pyStr(v: unknown): string {
  if (v === null || v === undefined) return "None";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") return isInt(v) ? String(v) : Number.isFinite(v) ? pyFloat(v) : Number.isNaN(v) ? "nan" : v > 0 ? "inf" : "-inf";
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return "[" + v.map(pyRepr).join(", ") + "]";
  return "{" + Object.entries(v as object).map(([k, x]) => `${pyReprStr(k)}: ${pyRepr(x)}`).join(", ") + "}";
}

/** Python truthiness of a JSON value. */
export const pyTruthy = (v: unknown) =>
  !(v === null || v === undefined || v === false || v === 0 || v === "" || (Array.isArray(v) && !v.length) || (typeof v === "object" && !Array.isArray(v) && !Object.keys(v as object).length));
