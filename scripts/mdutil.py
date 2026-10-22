"""Small markdown helpers shared by the build/audit scripts."""


def split_sections(text: str):
    """Split on level-2 headings ('## ') that are NOT inside fenced code blocks.

    Returns [(heading_text, body_text)], where heading_text excludes the leading '## '.
    Content before the first heading is ignored.
    """
    sections, head, buf, fence = [], None, [], False
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = not fence
        if not fence and line.startswith("## "):
            if head is not None:
                sections.append((head, "\n".join(buf)))
            head, buf = line[3:].strip(), []
        elif head is not None:
            buf.append(line)
    if head is not None:
        sections.append((head, "\n".join(buf)))
    return sections
