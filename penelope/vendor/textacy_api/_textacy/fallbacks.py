import re

from penelope import utility as pu

"""
Copyright 2016 Chartbeat, Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

(code modified)
"""
RE_LINEBREAK: re.Pattern = re.compile(r"(\r\n|[\n\v])+")
RE_NONBREAKING_SPACE: re.Pattern = re.compile(r"[^\S\n\v]+")
RE_ZWSP: re.Pattern = re.compile(r"[\u200B\u2060\uFEFF]+")


def whitespace(text: str) -> str:
    text = RE_ZWSP.sub("", text)
    text = RE_LINEBREAK.sub(r"\n", text)
    text = RE_NONBREAKING_SPACE.sub(" ", text)
    return text.strip()


# end copyright notice


class Corpus(pu.DummyClass):
    ...


class TopicModel(pu.DummyClass):
    ...


class Vectorizer(pu.DummyClass):
    ...


normalize_whitespace = whitespace
filter_terms_by_df = pu.create_dummy_function(tuple())
