"""
Thread-safe skill registry with indexing.

Follows NodePluginSystem pattern from graph/builder.py.
"""

import re
import logging
from typing import Dict, List, Optional, Iterator
from threading import RLock

from spoon_ai.skills.models import Skill, SkillTrigger

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    Thread-safe registry for skills with fast trigger matching.

    Maintains indexes for:
    - Tags: O(1) lookup by tag
    - Keywords: O(1) lookup by keyword
    - Intents: O(1) lookup by intent category
    - Patterns: Compiled regex patterns for matching
    """

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._lock = RLock()

        # Indexes for fast lookup
        self._tag_index: Dict[str, List[str]] = {}
        self._keyword_index: Dict[str, List[str]] = {}
        self._intent_index: Dict[str, List[str]] = {}
        self._pattern_cache: Dict[str, List[re.Pattern]] = {}

    def register(self, skill: Skill) -> None:
        """
        Register a skill and update indexes.

        Args:
            skill: Skill to register
        """
        with self._lock:
            name = skill.metadata.name

            if name in self._skills:
                logger.warning(f"Overwriting existing skill: {name}")
                # Clean up old indexes
                self._remove_from_indexes(name)

            self._skills[name] = skill
            self._index_skill(skill)
            logger.debug(f"Registered skill: {name}")

    def _index_skill(self, skill: Skill) -> None:
        """Build indexes for fast trigger matching."""
        name = skill.metadata.name

        # Tag index
        for tag in skill.metadata.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = []
            if name not in self._tag_index[tag_lower]:
                self._tag_index[tag_lower].append(name)

        # Trigger indexes
        for trigger in skill.metadata.triggers:
            if trigger.type == "keyword":
                for kw in trigger.keywords:
                    kw_lower = kw.lower()
                    if kw_lower not in self._keyword_index:
                        self._keyword_index[kw_lower] = []
                    if name not in self._keyword_index[kw_lower]:
                        self._keyword_index[kw_lower].append(name)

            elif trigger.type == "pattern":
                patterns = []
                for p in trigger.patterns:
                    try:
                        patterns.append(re.compile(p, re.IGNORECASE))
                    except re.error as e:
                        logger.warning(f"Invalid regex pattern in skill {name}: {p} - {e}")
                if patterns:
                    self._pattern_cache[name] = patterns

            elif trigger.type == "intent" and trigger.intent_category:
                category = trigger.intent_category.lower()
                if category not in self._intent_index:
                    self._intent_index[category] = []
                if name not in self._intent_index[category]:
                    self._intent_index[category].append(name)

    def _remove_from_indexes(self, name: str) -> None:
        """Remove a skill from all indexes."""
        # Remove from tag index
        for tag, names in list(self._tag_index.items()):
            if name in names:
                names.remove(name)
                if not names:
                    del self._tag_index[tag]

        # Remove from keyword index
        for kw, names in list(self._keyword_index.items()):
            if name in names:
                names.remove(name)
                if not names:
                    del self._keyword_index[kw]

        # Remove from intent index
        for intent, names in list(self._intent_index.items()):
            if name in names:
                names.remove(name)
                if not names:
                    del self._intent_index[intent]

        # Remove from pattern cache
        if name in self._pattern_cache:
            del self._pattern_cache[name]

    def unregister(self, name: str) -> bool:
        """
        Remove a skill from the registry.

        Args:
            name: Skill name to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name not in self._skills:
                return False

            self._remove_from_indexes(name)
            del self._skills[name]
            logger.debug(f"Unregistered skill: {name}")
            return True

    def get(self, name: str) -> Optional[Skill]:
        """
        Get a skill by name.

        Args:
            name: Skill name

        Returns:
            Skill or None if not found
        """
        with self._lock:
            return self._skills.get(name)

    def list_names(self) -> List[str]:
        """Get all registered skill names."""
        with self._lock:
            return list(self._skills.keys())

    def list_skills(self) -> List[Skill]:
        """Get all registered skills."""
        with self._lock:
            return list(self._skills.values())

    def find_by_tag(self, tag: str) -> List[Skill]:
        """
        Find skills by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching skills
        """
        with self._lock:
            names = self._tag_index.get(tag.lower(), [])
            return [self._skills[n] for n in names if n in self._skills]

    def find_by_keyword(self, text: str) -> List[Skill]:
        """
        Find skills by keyword matching.

        Extracts words from text and matches against keyword index.

        Args:
            text: Text to match keywords against

        Returns:
            List of matching skills (deduplicated)
        """
        with self._lock:
            words = set(re.findall(r'\w+', text.lower()))
            matched_names = set()

            for word in words:
                if word in self._keyword_index:
                    matched_names.update(self._keyword_index[word])

            return [self._skills[n] for n in matched_names if n in self._skills]

    def find_by_pattern(self, text: str) -> List[Skill]:
        """
        Find skills by regex pattern matching.

        Args:
            text: Text to match patterns against

        Returns:
            List of matching skills
        """
        with self._lock:
            matched = []

            for name, patterns in self._pattern_cache.items():
                if name not in self._skills:
                    continue

                for pattern in patterns:
                    if pattern.search(text):
                        matched.append(self._skills[name])
                        break  # Only add once per skill

            return matched

    def find_by_intent(self, intent_category: str) -> List[Skill]:
        """
        Find skills by intent category.

        Args:
            intent_category: Intent category to match

        Returns:
            List of matching skills
        """
        with self._lock:
            names = self._intent_index.get(intent_category.lower(), [])
            return [self._skills[n] for n in names if n in self._skills]

    def find_all_matching(self, text: str) -> List[Skill]:
        """
        Find all skills matching by keywords or patterns.

        Combines keyword and pattern matching, sorted by trigger priority.

        Args:
            text: Text to match against

        Returns:
            List of matching skills, sorted by priority (highest first)
        """
        with self._lock:
            keyword_matches = self.find_by_keyword(text)
            pattern_matches = self.find_by_pattern(text)

            # Combine and deduplicate
            seen = set()
            results = []

            for skill in keyword_matches + pattern_matches:
                if skill.metadata.name not in seen:
                    seen.add(skill.metadata.name)
                    results.append(skill)

            # Sort by priority (highest first)
            def get_priority(s: Skill) -> int:
                if s.metadata.triggers:
                    return max(t.priority for t in s.metadata.triggers)
                return 0

            results.sort(key=get_priority, reverse=True)
            return results

    def get_intent_categories(self) -> List[str]:
        """Get all registered intent categories."""
        with self._lock:
            return list(self._intent_index.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._skills)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._skills

    def __iter__(self) -> Iterator[Skill]:
        with self._lock:
            return iter(list(self._skills.values()))
