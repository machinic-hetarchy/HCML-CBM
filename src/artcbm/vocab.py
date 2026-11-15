"""Mine 'native' concept vocabularies from what annotators wrote, per language."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words about the act of looking or the artwork itself, plus function words the general
# stop lists miss. Kept short and visible on purpose.
EXTRA_STOP = {
    "english": {
        "painting",
        "picture",
        "image",
        "artist",
        "art",
        "looks",
        "look",
        "like",
        "feel",
        "feels",
        "feeling",
        "makes",
        "make",
        "really",
        "seems",
        "just",
        "think",
        "way",
        "lot",
        "bit",
        "don",
        "doesn",
        "isn",
        "it's",
        "i'm",
        "kind",
        "sort",
        "thing",
        "things",
        "piece",
        "work",
        "drawing",
        "drawn",
        "painted",
        "made",
        "reminds",
        "appears",
        "gives",
        "going",
        "quite",
        "looking",
        "little",
        "time",
    },
    "arabic": {
        "في",
        "من",
        "على",
        "إلى",
        "الى",
        "عن",
        "مع",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "التي",
        "الذي",
        "الذين",
        "هو",
        "هي",
        "هم",
        "انه",
        "أنه",
        "ان",
        "أن",
        "إن",
        "كان",
        "كانت",
        "يكون",
        "تكون",
        "لا",
        "ما",
        "لم",
        "لن",
        "قد",
        "كل",
        "بعض",
        "او",
        "أو",
        "ثم",
        "بين",
        "عند",
        "حتى",
        "وهو",
        "وهي",
        "مثل",
        "جدا",
        "جداً",
        "يبدو",
        "تبدو",
        "وكأن",
        "كأن",
        "وكأنه",
        "كأنه",
        "وكانه",
        "لوحة",
        "اللوحة",
        "لوحه",
        "اللوحه",
        "صورة",
        "الصورة",
        "الصوره",
        "رسم",
        "الرسم",
        "فنان",
        "الفنان",
        "يوجد",
        "توجد",
        "هناك",
        "فيها",
        "فيه",
        "عليها",
        "عليه",
        "منها",
        "منه",
        "به",
        "بها",
        "لها",
        "له",
        "وفي",
        "ومن",
        "وعلى",
        "اشعر",
        "أشعر",
        "يشعر",
        "تشعر",
        "شعور",
        "بشكل",
        "يجعلني",
        "تجعلني",
        "عندما",
        "لان",
        "لأن",
        "اكثر",
        "أكثر",
        "غير",
        "شيء",
        "الشيء",
        "نوع",
        "وهذا",
        "وهذه",
        "ولكن",
        "لكن",
        "علي",
        "ذات",
        "وان",
        "كثير",
        "وسط",
    },
    "chinese": {
        "一个",
        "这个",
        "那个",
        "我们",
        "他们",
        "她们",
        "自己",
        "什么",
        "没有",
        "非常",
        "感觉",
        "让人",
        "给人",
        "看起来",
        "觉得",
        "一种",
        "有点",
        "一些",
        "好像",
        "似乎",
        "仿佛",
        "显得",
        "起来",
        "这幅",
        "画面",
        "画中",
        "一幅",
        "图片",
        "画作",
        "作品",
        "画家",
        "这种",
        "可以",
        "还是",
        "就是",
        "但是",
        "因为",
        "所以",
        "如果",
        "而且",
        "虽然",
        "这样",
        "那么",
        "特别",
        "比较",
        "十分",
        "真的",
        "很多",
        "有些",
        "应该",
        "正在",
        "已经",
        "不是",
        "不会",
        "一下",
        "似的",
        "一起",
        "看到",
        "看着",
        "令人",
        "使人",
        "这是",
        "整个",
        "整幅",
        "上面",
        "里面",
        "旁边",
        "中间",
        "一样",
        "一片",
        "一位",
        "一只",
        "一张",
        "一座",
    },
}


def tokens(text: str, language: str) -> list[str]:
    """Content words of one explanation."""
    if language == "english":
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
        stop = ENGLISH_STOP_WORDS | EXTRA_STOP["english"]
        return [w for w in words if len(w) > 2 and w not in stop]
    if language == "arabic":
        out = []
        for w in re.findall(r"[ء-ي]+", text):
            if w in EXTRA_STOP["arabic"]:
                continue
            for prefix in ("وال", "بال", "ال"):  # fold 'the' / 'and the' / 'with the'
                if w.startswith(prefix) and len(w) > len(prefix) + 2:
                    w = w[len(prefix) :]
                    break
            if len(w) > 2 and w not in EXTRA_STOP["arabic"]:
                out.append(w)
        return out
    if language == "chinese":
        import logging

        import jieba

        jieba.setLogLevel(logging.WARNING)

        return [
            w
            for w in jieba.lcut(text)
            if len(w) >= 2 and re.fullmatch(r"[一-鿿]+", w) and w not in EXTRA_STOP["chinese"]
        ]
    raise ValueError(language)


def normalise(word: str, language: str) -> str:
    """Spelling variants that should count as one word (Arabic alef, taa marbuta, yaa)."""
    if language != "arabic":
        return word
    return re.sub("[أإآ]", "ا", word).replace("ة", "ه").replace("ى", "ي")


def mine_vocabulary(captions: pd.Series, language: str, k: int, with_counts: bool = False):
    """The k words that appear in the most explanations (each explanation counts once).
    Spelling variants are counted together and shown in their most common spelling.
    With `with_counts`, also returns how many explanations contain each word."""
    counts, spellings = Counter(), {}
    for text in captions:
        words = tokens(str(text), language)
        counts.update({normalise(w, language) for w in words})
        for w in words:
            spellings.setdefault(normalise(w, language), Counter())[w] += 1
    top = counts.most_common(k)
    words = [spellings[w].most_common(1)[0][0] for w, _ in top]
    return (words, [n for _, n in top]) if with_counts else words
