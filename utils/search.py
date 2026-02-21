"""搜索工具：简繁转换、拼音匹配"""
_opencc_cache: dict[str, object] = {}


def _get_opencc(preset: str):
    """懒加载 OpenCC 转换器，preset: 's2t' | 't2s'"""
    if preset not in _opencc_cache:
        try:
            from opencc import OpenCC
            _opencc_cache[preset] = OpenCC(preset)
        except Exception:
            _opencc_cache[preset] = False
    return _opencc_cache[preset]


def to_simplified(s: str) -> str:
    """转为简体（用于匹配）"""
    cc = _get_opencc("t2s")
    if not cc:
        return s
    try:
        return cc.convert(s)
    except Exception:
        return s


def to_traditional(s: str) -> str:
    """转为繁体（用于匹配）"""
    cc = _get_opencc("s2t")
    if not cc:
        return s
    try:
        return cc.convert(s)
    except Exception:
        return s


def to_pinyin_lower(s: str) -> str:
    """将中文转为小写无声调拼音并拼接（如 厦门 -> xiamen），非中文保留原样并转小写。"""
    try:
        from pypinyin import lazy_pinyin, Style
        parts = lazy_pinyin(s, style=Style.NORMAL)
        return "".join(p).lower() if (p := [x.strip() for x in parts if x]) else s.lower()
    except Exception:
        return s.lower()


def search_match(query: str, target: str) -> bool:
    """判断 query 是否匹配 target：支持模糊、简繁、拼音。"""
    if not query or not target:
        return False
    q = query.strip()
    t = target
    if q.lower() in t.lower():
        return True
    try:
        q_s, q_t = to_simplified(q), to_traditional(q)
        t_s, t_t = to_simplified(t), to_traditional(t)
        if q_s and q_s in t_s:
            return True
        if q_t and q_t in t_t:
            return True
    except Exception:
        pass
    try:
        t_py = to_pinyin_lower(t)
        q_lower = q.lower()
        if q_lower in t_py:
            return True
        if any("\u4e00" <= c <= "\u9fff" for c in q):
            q_py = to_pinyin_lower(q)
            if q_py and q_py in t_py:
                return True
    except Exception:
        pass
    return False
