import re
with open("05_Phase4/README.md", "r", encoding="utf-8") as f:
    c = f.read().lstrip("\ufeff")
# Each mojibake group encodes utf-8 bytes as cp1252 codepoints
# Fix: re-encode as cp1252 bytes, decode as utf-8
def fix_mojibake(s):
    result = []
    i = 0
    while i < len(s):
        # try to encode a window as cp1252 and decode as utf-8
        found = False
        for w in (4, 3, 2):
            chunk = s[i:i+w]
            try:
                b = chunk.encode("cp1252")
                decoded = b.decode("utf-8")
                if len(decoded) == 1 and ord(decoded) > 127:
                    result.append(decoded)
                    i += w
                    found = True
                    break
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        if not found:
            result.append(s[i])
            i += 1
    return "".join(result)
fixed = fix_mojibake(c)
with open("05_Phase4/README.md", "w", encoding="utf-8") as f:
    f.write(fixed)
hits = set(re.findall(r"[^\x00-\x7F]+", fixed))
print("Remaining non-ASCII:", hits if hits else "NONE (all fixed)")
print("Lines:", fixed.count("\n"))
