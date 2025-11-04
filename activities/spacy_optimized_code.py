
 ## OPTIMIZED CODE : 50% Relevance : SUMMARY , DESCRIPTION and COMMENTS


import json



import spacy
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter, defaultdict
import argparse
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time


class SemanticErrorAnalyzer:
    """
    Advanced error analyzer that uses NLP to understand error context
    and extract the most meaningful, business-focused information.
    """

    def __init__(self, nlp):
        self.nlp = nlp

        # Define semantic categories for errors
        self.error_categories = {
            'device_state': ['offline', 'unresponsive', 'disconnected', 'unavailable',
                             'powered off'],
            'timeout': ['timeout', 'timed out', 'waiting', 'expired', 'no response'],
            'validation': ['greater', 'less', 'equal', 'mismatch', 'incorrect', 'invalid',
                           'expected', 'actual'],
            'detection': ['not detected', 'failed to detect', 'missing', 'not found',
                          'unable to find'],
            'communication': ['failed', 'error', 'connection', 'request', 'response',
                              'status code'],
            'ui_issue': ['not showing', 'not displayed', 'incorrect display', 'ui issue', 'visual'],
        }

        # Business-level error templates
        self.error_templates = {
            'device_unresponsive': 'Device is unresponsive or offline',
            'timeout_operation': 'Operation timed out',
            'validation_failed': '{subject} {comparison} expected value',
            'detection_failed': '{feature} not detected',
            'command_failed': 'Command failed: {reason}',
            'setup_failed': 'Test setup failed: {reason}',
        }

    def analyze_error(self, block: str, test_description: str = '') -> Dict[str, any]:
        """
        Analyze error block using NLP to understand context and extract meaningful information.
        """

        # Step 1: Extract error sentences
        error_sentences = self._extract_error_sentences(block)

        if not error_sentences:
            return {
                'error': test_description or 'Test failed - no error details',
                'category': 'unknown',
                'root_cause': 'unknown',
                'confidence': 'low',
                'technical_details': ''
            }

        # Step 2: Analyze each sentence semantically
        analyzed_sentences = []
        for sentence in error_sentences:
            analysis = self._analyze_sentence(sentence)
            if analysis:
                analyzed_sentences.append(analysis)

        # Step 3: Find the most meaningful error
        best_error = self._select_best_error(analyzed_sentences, test_description)

        return best_error

    def _extract_error_sentences(self, block: str) -> List[str]:
        """Extract potential error sentences from the block."""
        sentences = []

        # Look for specific error patterns
        patterns = [
            r'Failure Message:\s*(.+?)(?:\nFailure hash:|$)',
            r'\{"reason"\s*:\s*"([^"]+)"\s*,\s*"errorCode"\s*:\s*"([^"]+)"\}',
            r'CommandFailed:\s*(.+?)(?:\n|$)',
            r'AssertionError:\s*(.+?)(?:\n|$)',
            r'TimeoutError:\s*(.+?)(?:\n|$)',
            r'DeviceIsOfflineError:\s*(.+?)(?:\n|$)',
            r'(\w+Error):\s*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, block, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if match.lastindex == 2:
                    sentence = f"{match.group(1)}: {match.group(2)}"
                else:
                    sentence = match.group(1) if match.lastindex >= 1 else match.group(0)

                sentence = sentence.strip()
                if len(sentence) > 10 and sentence not in sentences:
                    sentences.append(sentence)

        return sentences[:10]

    def _analyze_sentence(self, sentence: str) -> Optional[Dict]:
        """Analyze a sentence using spaCy to understand its semantic meaning."""

        cleaned = self._preliminary_clean(sentence)

        if len(cleaned) < 10:
            return None

        doc = self.nlp(cleaned.lower())

        analysis = {
            'original': sentence,
            'cleaned': cleaned,
            'tokens': [],
            'entities': [],
            'root_verb': None,
            'negations': [],
            'subjects': [],
            'objects': [],
            'adjectives': [],
            'category': None,
            'score': 0
        }

        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop:
                analysis['tokens'].append(token.lemma_)

            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                analysis['root_verb'] = token.lemma_

            if token.dep_ == 'neg':
                analysis['negations'].append(token.head.lemma_)

            if token.dep_ in ['nsubj', 'nsubjpass']:
                analysis['subjects'].append(token.lemma_)

            if token.dep_ in ['dobj', 'pobj']:
                analysis['objects'].append(token.lemma_)

            if token.pos_ == 'ADJ':
                analysis['adjectives'].append(token.lemma_)

        for ent in doc.ents:
            analysis['entities'].append({
                'text': ent.text,
                'label': ent.label_
            })

        analysis['category'] = self._categorize_error(analysis)
        analysis['score'] = self._score_error_analysis(analysis)

        return analysis

    def _preliminary_clean(self, text: str) -> str:
        """Do basic cleaning before NLP analysis."""
        text = re.sub(r'\(ID=[a-f0-9-]+\)', '', text)
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'File\s+"[^"]+"', '', text)
        text = re.sub(r"'deviceId'\s*:\s*'[^']+'", '', text)
        text = re.sub(r"with data \{[^}]+\}", '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _categorize_error(self, analysis: Dict) -> str:
        """Categorize error based on semantic analysis."""
        tokens_str = ' '.join(analysis['tokens']).lower()

        for category, keywords in self.error_categories.items():
            if any(keyword in tokens_str for keyword in keywords):
                return category

        if analysis['root_verb']:
            if analysis['root_verb'] in ['fail', 'timeout', 'expire']:
                return 'timeout' if 'timeout' in tokens_str else 'communication'
            elif analysis['root_verb'] in ['detect', 'find', 'see']:
                return 'detection'
            elif analysis['root_verb'] in ['match', 'equal', 'compare']:
                return 'validation'

        return 'unknown'

    def _score_error_analysis(self, analysis: Dict) -> float:
        """Score how meaningful/business-focused the error is."""
        score = 0.0

        category_scores = {
            'validation': 10,
            'detection': 9,
            'device_state': 8,
            'ui_issue': 8,
            'timeout': 6,
            'communication': 5,
            'unknown': 2
        }
        score += category_scores.get(analysis['category'], 0)

        if analysis['subjects']:
            score += 2
        if analysis['objects']:
            score += 2

        score += len(analysis['adjectives']) * 0.5
        score += len(analysis['negations']) * 1.5

        technical_terms = ['request', 'command', 'send', 'call', 'method', 'function']
        technical_count = sum(1 for token in analysis['tokens'] if token in technical_terms)
        score -= technical_count * 0.5

        if len(analysis['cleaned']) > 200:
            score -= 2

        return max(0, score)

    def _select_best_error(self, analyzed_sentences: List[Dict], test_description: str) -> Dict:
        """Select the best error from analyzed sentences."""

        if not analyzed_sentences:
            return {
                'error': test_description or 'Test failed',
                'category': 'unknown',
                'root_cause': 'unknown',
                'confidence': 'low',
                'technical_details': ''
            }

        analyzed_sentences.sort(key=lambda x: x['score'], reverse=True)
        best = analyzed_sentences[0]

        error_message = self._generate_error_message(best, test_description)
        root_cause = self._extract_root_cause(best)
        confidence = 'high' if best['score'] >= 8 else 'medium' if best['score'] >= 5 else 'low'

        return {
            'error': error_message,
            'category': best['category'],
            'root_cause': root_cause,
            'confidence': confidence,
            'technical_details': best['cleaned'][:200],
            'semantic_analysis': {
                'root_verb': best['root_verb'],
                'subjects': best['subjects'],
                'objects': best['objects'],
                'negations': best['negations']
            }
        }

    def _generate_error_message(self, analysis: Dict, test_description: str) -> str:
        """Generate a human-readable error message from semantic analysis."""

        category = analysis['category']

        if category == 'device_state':
            if 'unresponsive' in analysis['tokens'] or 'offline' in analysis['tokens']:
                return 'Device is unresponsive or offline'
            return 'Device state error'

        if category == 'timeout':
            if analysis['objects']:
                obj = analysis['objects'][0]
                return f'Operation timed out waiting for {obj}'
            return 'Operation timed out'

        if category == 'validation':
            comparisons = ['greater', 'less', 'equal', 'mismatch']
            found_comparison = None
            for comp in comparisons:
                if comp in analysis['tokens']:
                    found_comparison = comp
                    break

            if found_comparison and analysis['subjects']:
                subject = analysis['subjects'][0].replace('_', ' ')
                return f'{subject.title()} is {found_comparison} than expected'

            if analysis['adjectives']:
                adj = analysis['adjectives'][0]
                return f'Validation failed: value is {adj}'

            return 'Validation failed'

        if category == 'detection':
            if analysis['negations']:
                negated = analysis['negations'][0].replace('_', ' ')
                return f'{negated.title()} not detected'

            if analysis['objects']:
                obj = analysis['objects'][0].replace('_', ' ')
                return f'{obj.title()} not detected'

            return 'Detection failed'

        if category == 'ui_issue':
            if analysis['objects']:
                obj = analysis['objects'][0].replace('_', ' ')
                return f'UI Issue: {obj} not displayed correctly'
            return 'UI display issue'

        if category == 'communication':
            status_match = re.search(r'status code (\d+)', analysis['cleaned'])
            if status_match:
                status = status_match.group(1)
                return f'Communication error (HTTP {status})'

            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', analysis['original'])
            if reason_match:
                reason = reason_match.group(1).replace('_', ' ').title()
                return f'Communication error: {reason}'

            return 'Communication error'

        cleaned = analysis['cleaned']
        if len(cleaned) > 100:
            if ':' in cleaned:
                parts = cleaned.split(':')
                cleaned = parts[-1].strip()

        return cleaned[:150]

    def _extract_root_cause(self, analysis: Dict) -> str:
        """Extract the root cause from semantic analysis."""

        error_code_match = re.search(r'"errorCode"\s*:\s*"([^"]+)"', analysis['original'])
        if error_code_match:
            return error_code_match.group(1).replace('_', ' ')

        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', analysis['original'])
        if reason_match:
            return reason_match.group(1).replace('_', ' ')

        if analysis['root_verb']:
            return f"{analysis['category']}: {analysis['root_verb']}"

        return analysis['category']


class OptimizedJiraTicketMatcher:
    """Optimized matcher with pre-computed embeddings and fast filtering."""

    def __init__(self, model_name: str = "en_core_web_md"):
        """Initialize the matcher with a spaCy model."""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

        self.error_analyzer = SemanticErrorAnalyzer(self.nlp)

        # Cache for embeddings
        self.jira_embeddings_cache = {}
        self.jira_keywords_cache = {}

    def precompute_jira_embeddings(self, jira_tickets: List[Dict], include_comments: bool = True,
                                   max_comments: int = 10):
        """
        Pre-compute embeddings for all JIRA tickets.
        This is done once and dramatically speeds up matching.
        """
        print("🚀 Pre-computing JIRA ticket embeddings...")
        start_time = time.time()

        for idx, ticket in enumerate(jira_tickets):
            jira_key = ticket.get('key', f'ticket_{idx}')

            # Extract text
            jira_data = self.extract_jira_text(ticket, include_comments, max_comments)

            # Compute embeddings for different fields
            embeddings = {}

            if jira_data.get('summary'):
                embeddings['summary'] = self.nlp(jira_data['summary']).vector

            if jira_data.get('description'):
                embeddings['description'] = self.nlp(jira_data['description']).vector

            if jira_data.get('comments'):
                embeddings['comments'] = self.nlp(jira_data['comments']).vector

            if jira_data.get('full_text'):
                embeddings['full_text'] = self.nlp(jira_data['full_text']).vector

            # Store in cache
            self.jira_embeddings_cache[jira_key] = {
                'embeddings': embeddings,
                'data': jira_data,
                'ticket': ticket
            }

            # Pre-compute keywords
            self.jira_keywords_cache[jira_key] = self.extract_keywords(jira_data['full_text'])

            if (idx + 1) % 50 == 0:
                print(f"   Processed {idx + 1}/{len(jira_tickets)} tickets...")

        elapsed = time.time() - start_time
        print(f"✅ Pre-computed embeddings for {len(jira_tickets)} tickets in {elapsed:.2f}s")

    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text using spaCy."""
        doc = self.nlp(text.lower())

        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB'] and
                    not token.is_stop and
                    len(token.text) > 2):
                keywords.append(token.lemma_)

        return keywords

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append(ent.text.lower())
        return entities

    def extract_jira_text(self, jira_ticket: Dict, include_comments: bool = True,
                          max_comments: int = 10) -> Dict[str, str]:
        """Extract relevant text from Jira ticket for comparison."""
        fields = jira_ticket.get('fields', {})

        result = {
            'summary': '',
            'description': '',
            'labels': '',
            'comments': '',
            'full_text': '',
            'status': '',
            'created': '',
            'updated': ''
        }

        # Add summary
        summary = fields.get('summary', '')
        if summary and len(summary.strip()) > 0:
            result['summary'] = summary.strip()

        # Add description
        description = fields.get('description', '')
        if description and len(description.strip()) > 0:
            description_clean = re.sub(r'https?://\S+', '', description)
            description_clean = re.sub(r'\*+', '', description_clean)
            description_clean = re.sub(r'\|[^\|]+\|', '', description_clean)
            description_clean = re.sub(r'\s+', ' ', description_clean)
            description_clean = description_clean.strip()
            if len(description_clean) > 10:
                result['description'] = description_clean[:2000]

        # Add labels
        labels = fields.get('labels', [])
        if labels:
            result['labels'] = ' '.join(labels)

        # Add status
        status_obj = fields.get('status', {})
        if status_obj:
            result['status'] = status_obj.get('name', '').lower()

        # Add timestamps
        result['created'] = fields.get('created', '')
        result['updated'] = fields.get('updated', '')

        # Add comments
        if include_comments:
            comments_data = fields.get('comment', {})
            if comments_data:
                comments_list = comments_data.get('comments', [])
                recent_comments = comments_list[-max_comments:] if len(
                    comments_list) > max_comments else comments_list

                comment_texts = []
                for comment in recent_comments:
                    comment_body = comment.get('body', '')
                    if comment_body and len(comment_body.strip()) > 10:
                        comment_clean = re.sub(r'https?://\S+', '', comment_body)
                        comment_clean = re.sub(r'\*+', '', comment_clean)
                        comment_clean = re.sub(r'\s+', ' ', comment_clean)
                        comment_clean = comment_clean.strip()
                        if len(comment_clean) > 10:
                            comment_texts.append(comment_clean[:500])

                if comment_texts:
                    result['comments'] = ' '.join(comment_texts)

        # Combine all parts
        text_parts = [
            result['summary'],
            result['summary'],
            result['description'],
            result['comments'],
            result['labels']
        ]
        result['full_text'] = ' '.join([t for t in text_parts if t]).strip()

        return result

    def calculate_keyword_overlap(self, test_keywords: List[str],
                                  jira_keywords: List[str]) -> float:
        """Calculate keyword overlap score."""
        if not test_keywords or not jira_keywords:
            return 0.0

        test_set = set(test_keywords)
        jira_set = set(jira_keywords)

        intersection = test_set.intersection(jira_set)
        union = test_set.union(jira_set)

        return len(intersection) / len(union) if union else 0.0

    def calculate_fast_semantic_similarity(self, test_vector: np.ndarray, jira_vector: np.ndarray) -> float:
        """
        Fast semantic similarity using pre-computed vectors.
        Uses cosine similarity.
        """
        if test_vector is None or jira_vector is None:
            return 0.0

        # Check if vectors have norm
        test_norm = np.linalg.norm(test_vector)
        jira_norm = np.linalg.norm(jira_vector)

        if test_norm == 0 or jira_norm == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(test_vector, jira_vector) / (test_norm * jira_norm)

        return max(0.0, float(similarity))

    def calculate_exact_match_bonus(self, test_info: Dict, jira_data: Dict,
                                    jira_ticket: Dict) -> float:
        """Calculate bonus score for exact or near-exact matches."""
        bonus = 0.0

        # 1. Exact error message match
        test_error = test_info.get('error', '').lower()
        jira_summary = jira_data.get('summary', '').lower()
        jira_description = jira_data.get('description', '').lower()

        if test_error and len(test_error) > 20:
            if test_error in jira_summary or jira_summary in test_error:
                bonus += 0.25
            elif test_error in jira_description or any(
                    test_error in sent for sent in jira_description.split('.')):
                bonus += 0.15

        # 2. Component/path similarity
        test_component = test_info.get('component', '').lower()
        if test_component:
            component_parts = set(test_component.split())
            jira_text_lower = jira_data.get('full_text', '').lower()

            matching_components = sum(
                1 for part in component_parts if part in jira_text_lower and len(part) > 3)
            if matching_components > 0:
                bonus += min(0.10, matching_components * 0.03)

        # 3. Error category match
        test_category = test_info.get('category', '')
        if test_category and test_category != 'unknown':
            category_keywords = {
                'device_state': ['device', 'offline', 'unresponsive'],
                'timeout': ['timeout', 'timed out'],
                'validation': ['validation', 'assert', 'expected', 'actual'],
                'detection': ['detect', 'not found', 'missing'],
                'ui_issue': ['ui', 'display', 'visual', 'showing'],
                'communication': ['communication', 'connection', 'request', 'response']
            }

            if test_category in category_keywords:
                jira_lower = jira_data.get('full_text', '').lower()
                category_matches = sum(
                    1 for kw in category_keywords[test_category] if kw in jira_lower)
                if category_matches >= 2:
                    bonus += 0.08

        # 4. Root cause match
        test_root_cause = test_info.get('root_cause', '').lower()
        if test_root_cause and test_root_cause != 'unknown':
            if test_root_cause in jira_data.get('full_text', '').lower():
                bonus += 0.10

        # 5. Test case ID match
        test_case_id = test_info.get('test_case_id', '')
        if test_case_id:
            if test_case_id in jira_data.get('full_text', ''):
                bonus += 0.30

        # 6. Entity matching
        test_entities = self.extract_entities(test_info.get('full_text', ''))
        jira_entities = self.extract_entities(jira_data.get('full_text', ''))

        if test_entities and jira_entities:
            common_entities = set(test_entities).intersection(set(jira_entities))
            if common_entities:
                bonus += min(0.12, len(common_entities) * 0.04)

        return min(bonus, 0.50)

    def calculate_status_relevance_bonus(self, jira_data: Dict) -> float:
        """Calculate bonus based on JIRA ticket status."""
        status = jira_data.get('status', '').lower()

        status_bonuses = {
            'open': 0.10,
            'in progress': 0.08,
            'reopened': 0.09,
            'to do': 0.07,
            'in review': 0.06,
            'resolved': -0.05,
            'closed': -0.08,
            'done': -0.05,
            'cancelled': -0.10
        }

        return status_bonuses.get(status, 0.0)

    def calculate_recency_bonus(self, jira_data: Dict) -> float:
        """Calculate bonus based on how recent the ticket is."""
        try:
            updated = jira_data.get('updated', '')
            if not updated:
                return 0.0

            updated_date = datetime.fromisoformat(updated.replace('Z', '+00:00'))
            now = datetime.now(updated_date.tzinfo)
            days_old = (now - updated_date).days

            if days_old < 7:
                return 0.08
            elif days_old < 30:
                return 0.05
            elif days_old < 90:
                return 0.02
            elif days_old > 365:
                return -0.03
            else:
                return 0.0
        except:
            return 0.0

    def calculate_confidence_weight(self, test_info: Dict) -> float:
        """Adjust scoring based on error analysis confidence."""
        confidence = test_info.get('confidence', 'low')

        confidence_multipliers = {
            'high': 1.15,
            'medium': 1.05,
            'low': 0.95
        }

        return confidence_multipliers.get(confidence, 1.0)

    def calculate_optimized_multi_field_score(
            self,
            test_info: Dict[str, str],
            test_vector: np.ndarray,
            test_keywords: List[str],
            jira_key: str,
            summary_weight: float = 0.40,
            description_weight: float = 0.25,
            comment_weight: float = 0.15,
            keyword_weight: float = 0.20
    ) -> Dict[str, float]:
        """
        Calculate enhanced similarity score using pre-computed embeddings.
        MUCH FASTER than original method.
        """

        # Get cached data
        cached = self.jira_embeddings_cache.get(jira_key)
        if not cached:
            return None

        jira_embeddings = cached['embeddings']
        jira_data = cached['data']
        jira_ticket = cached['ticket']
        jira_keywords = self.jira_keywords_cache.get(jira_key, [])

        # Fast similarity calculations using pre-computed vectors
        summary_score = 0.0
        if 'summary' in jira_embeddings:
            summary_score = self.calculate_fast_semantic_similarity(test_vector, jira_embeddings['summary'])

        description_score = 0.0
        if 'description' in jira_embeddings:
            description_score = self.calculate_fast_semantic_similarity(test_vector, jira_embeddings['description'])

        comment_score = 0.0
        if 'comments' in jira_embeddings:
            comment_score = self.calculate_fast_semantic_similarity(test_vector, jira_embeddings['comments'])

        keyword_score = self.calculate_keyword_overlap(test_keywords, jira_keywords)

        # Base weighted score
        base_score = (
                summary_weight * summary_score +
                description_weight * description_score +
                comment_weight * comment_score +
                keyword_weight * keyword_score
        )

        # Apply confidence multiplier
        confidence_multiplier = self.calculate_confidence_weight(test_info)
        adjusted_score = base_score * confidence_multiplier

        # Calculate bonuses
        exact_match_bonus = self.calculate_exact_match_bonus(test_info, jira_data, jira_ticket)
        status_bonus = self.calculate_status_relevance_bonus(jira_data)
        recency_bonus = self.calculate_recency_bonus(jira_data)

        # Final score with bonuses
        final_score = adjusted_score + exact_match_bonus + status_bonus + recency_bonus

        # Cap at 1.0 (100%)
        final_score = min(final_score, 1.0)

        return {
            'multi_field': final_score,
            'base_score': base_score,
            'summary_score': summary_score,
            'description_score': description_score,
            'comment_score': comment_score,
            'keyword_score': keyword_score,
            'exact_match_bonus': exact_match_bonus,
            'status_bonus': status_bonus,
            'recency_bonus': recency_bonus,
            'confidence_multiplier': confidence_multiplier,
            'test_keywords': test_keywords[:10],
            'jira_keywords': jira_keywords[:10]
        }

    def find_matching_tickets(
            self,
            failed_tests: List[Dict],
            jira_tickets: List[Dict],
            top_n: Optional[int] = 5,
            threshold: float = 0.50,
            use_multi_field: bool = True,
            include_comments: bool = True,
            max_comments: int = 10
    ) -> List[Dict]:
        """Find matching Jira tickets for failed tests with optimized scoring."""

        # Pre-compute JIRA embeddings once
        if not self.jira_embeddings_cache:
            self.precompute_jira_embeddings(jira_tickets, include_comments, max_comments)

        results = []

        print(f"\n🔍 Matching {len(failed_tests)} tests against {len(jira_tickets)} tickets...")
        start_time = time.time()

        for test_idx, test_dict in enumerate(failed_tests):
            test_path = test_dict.get('test_path', '')
            error = test_dict.get('error', '')
            category = test_dict.get('category', 'unknown')
            root_cause = test_dict.get('root_cause', '')
            confidence = test_dict.get('confidence', 'low')
            test_case_id = test_dict.get('test_case_id', '')

            test_info = {
                'test_path': test_path,
                'test_name': test_path.split('::')[-1] if '::' in test_path else
                test_path.split('/')[-1] if '/' in test_path else test_path,
                'error': error,
                'category': category,
                'root_cause': root_cause,
                'confidence': confidence,
                'test_case_id': test_case_id,
                'component': ' '.join([p for p in test_path.split('/') if
                                       not p.endswith('.py')]) if '/' in test_path else '',
                'full_text': f"{test_path} {error} {root_cause} {test_case_id}"
            }

            test_name_clean = re.sub(r'test_', '', test_info['test_name'])
            test_name_clean = re.sub(r'_', ' ', test_name_clean)
            test_info['test_description'] = test_name_clean

            if not test_info.get('test_name'):
                continue

            # Pre-compute test vector and keywords once
            test_text_parts = [
                test_info.get('test_description', ''),
                test_info.get('error', ''),
                test_info.get('root_cause', ''),
                test_info.get('component', '')
            ]
            test_text = ' '.join([t for t in test_text_parts if t])
            test_vector = self.nlp(test_text).vector
            test_keywords = self.extract_keywords(test_text)

            similarities = []

            # Fast iteration through cached JIRA tickets
            for jira_key in self.jira_embeddings_cache.keys():
                scores = self.calculate_optimized_multi_field_score(
                    test_info,
                    test_vector,
                    test_keywords,
                    jira_key
                )

                if not scores:
                    continue

                score = scores['multi_field']

                if score >= threshold:
                    cached = self.jira_embeddings_cache[jira_key]
                    ticket = cached['ticket']

                    match_info = {
                        'ticket': ticket,
                        'similarity': score,
                        'jira_key': ticket.get('key', 'N/A'),
                        'summary': ticket.get('fields', {}).get('summary', 'N/A'),
                        'status': ticket.get('fields', {}).get('status', {}).get('name', 'N/A'),
                        'matching_keywords': list(
                            set(scores['test_keywords']).intersection(set(scores['jira_keywords'])))
                    }

                    # Add all score components
                    match_info['base_score'] = scores['base_score']
                    match_info['summary_score'] = scores['summary_score']
                    match_info['description_score'] = scores['description_score']
                    match_info['comment_score'] = scores.get('comment_score', 0.0)
                    match_info['keyword_score'] = scores['keyword_score']
                    match_info['exact_match_bonus'] = scores.get('exact_match_bonus', 0.0)
                    match_info['status_bonus'] = scores.get('status_bonus', 0.0)
                    match_info['recency_bonus'] = scores.get('recency_bonus', 0.0)
                    match_info['confidence_multiplier'] = scores.get('confidence_multiplier', 1.0)

                    similarities.append(match_info)

            similarities.sort(key=lambda x: x['similarity'], reverse=True)

            if top_n is None:
                top_matches = similarities
            else:
                top_matches = similarities[:top_n]

            result_entry = {
                'test_name': test_info.get('test_name'),
                'test_path': test_info.get('test_path', 'N/A'),
                'test_description': test_info.get('test_description', 'N/A'),
                'test_case_id': test_info.get('test_case_id', 'N/A'),
                'error': test_info.get('error', 'N/A'),
                'error_category': test_info.get('category', 'unknown'),
                'root_cause': test_info.get('root_cause', 'unknown'),
                'error_confidence': test_info.get('confidence', 'low'),
                'component': test_info.get('component', 'N/A'),
                'matches': top_matches
            }

            results.append(result_entry)

            if (test_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (test_idx + 1)
                remaining = avg_time * (len(failed_tests) - test_idx - 1)
                print(f"   Processed {test_idx + 1}/{len(failed_tests)} tests... "
                      f"(~{remaining:.1f}s remaining)")

        elapsed = time.time() - start_time
        print(f"✅ Matching completed in {elapsed:.2f}s ({elapsed / len(failed_tests):.2f}s per test)")

        return results

    def print_results(self, results: List[Dict], show_details: bool = True,
                      show_score_breakdown: bool = False):
        """Print formatted results with enhanced scoring details."""
        print("\n" + "=" * 80)
        print("JIRA TICKET MATCHING RESULTS (Optimized Enhanced Scoring)")
        print("=" * 80 + "\n")

        tests_with_matches = [r for r in results if r['matches']]
        tests_without_matches = [r for r in results if not r['matches']]

        print(f"Total Tests: {len(results)}")
        print(f"Tests with Matches: {len(tests_with_matches)}")
        print(f"Tests without Matches: {len(tests_without_matches)}")

        for idx, result in enumerate(tests_with_matches, 1):
            print(f"\n{'─' * 80}")
            print(f"TEST #{idx}: {result['test_name']}")
            print(f"{'─' * 80}")
            print(f"Path: {result['test_path']}")
            print(f"Description: {result['test_description']}")
            if result.get('test_case_id') and result['test_case_id'] != 'N/A':
                print(f"Test Case ID: {result['test_case_id']}")
            print(f"Component: {result['component']}")

            error_text = result['error']
            error_category = result.get('error_category', 'unknown')
            root_cause = result.get('root_cause', 'unknown')
            error_confidence = result.get('error_confidence', 'low')

            if error_text and error_text != 'N/A':
                print(f"\n🔍 Error Analysis:")
                print(f"   Message: {error_text}")
                print(f"   Category: {error_category}")
                print(f"   Root Cause: {root_cause}")
                print(f"   Confidence: {error_confidence}")
            else:
                print(f"Error: No error message found")

            print(f"\nMatching Jira Tickets:")

            for match_idx, match in enumerate(result['matches'], 1):
                print(
                    f"\n  {match_idx}. {match['jira_key']} - Overall Score: {match['similarity']:.2%}")
                print(f"     Status: {match['status']}")
                print(f"     Summary: {match['summary']}")

                if show_details:
                    print(f"     Summary: {match['summary_score']:.2%} | "
                          f"Description: {match['description_score']:.2%} | "
                          f"Comments: {match.get('comment_score', 0):.2%} | "
                          f"Keywords: {match['keyword_score']:.2%}")

                    if show_score_breakdown:
                        exact_bonus = match.get('exact_match_bonus', 0)
                        status_bonus = match.get('status_bonus', 0)
                        recency_bonus = match.get('recency_bonus', 0)

                        if exact_bonus > 0 or status_bonus != 0 or recency_bonus != 0:
                            print(f"     Bonuses: Exact Match: +{exact_bonus:.2%} | "
                                  f"Status: {status_bonus:+.2%} | "
                                  f"Recency: {recency_bonus:+.2%}")

                    if match['matching_keywords']:
                        print(
                            f"     Matching Keywords: {', '.join(match['matching_keywords'][:5])}")

        if tests_without_matches:
            print(f"\n{'=' * 80}")
            print(f"TESTS WITHOUT MATCHES (Below 50% Threshold)")
            print(f"{'=' * 80}\n")
            for idx, result in enumerate(tests_without_matches, 1):
                print(f"{idx}. {result['test_name']} - {result['test_description']}")
                if result['error'] and result['error'] != 'N/A':
                    error_preview = result['error'][:100]
                    print(f"   Error: {error_preview}...")
                    print(f"   Category: {result.get('error_category', 'unknown')}")

        print(f"\n{'=' * 80}\n")


def read_failed_tests_new_format(file_path: str) -> List[Dict[str, str]]:
    """Read and parse failed tests with semantic NLP error analysis."""
    failed_tests = []

    try:
        print("Initializing semantic analyzer...")
        nlp = spacy.load("en_core_web_md")
        error_analyzer = SemanticErrorAnalyzer(nlp)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        test_blocks = re.split(r'\n\n\n+', content)

        print(f"Analyzing {len(test_blocks)} test blocks...")

        for block in test_blocks:
            block = block.strip()

            if not block:
                continue

            lines = block.split('\n')
            first_line = lines[0] if lines else ''

            test_case_id_match = re.search(r'\(C\d+\)', first_line)
            test_case_id = test_case_id_match.group(0) if test_case_id_match else ''

            test_description = re.sub(r'\s*\(C\d+\)\s*$', '', first_line).strip()

            test_method_match = re.search(r'(?:ERROR|FAIL):\s+(\w+)\s+\([\w.]+\)', block)
            test_method_name = test_method_match.group(1) if test_method_match else ''

            test_path_match = re.search(r'File\s+"([^"]+automated_tests/[^"]+\.py)"', block)
            if test_path_match:
                test_path = test_path_match.group(1)
                test_path = test_path.replace('/workspaces/storage/automated_tests/', '')
                test_path = test_path.replace('/root/roku_automation/', '')
                if test_method_name:
                    test_path = f"{test_path}::{test_method_name}"
            else:
                test_path = test_method_name if test_method_name else test_description

            print(f"  Analyzing: {test_method_name or test_description[:50]}...")
            error_analysis = error_analyzer.analyze_error(block, test_description)

            if test_path or test_method_name:
                failed_tests.append({
                    'test_path': test_path,
                    'test_description': test_description,
                    'test_case_id': test_case_id,
                    'error': error_analysis['error'],
                    'category': error_analysis['category'],
                    'root_cause': error_analysis['root_cause'],
                    'confidence': error_analysis['confidence'],
                    'technical_details': error_analysis.get('technical_details', ''),
                    'semantic_analysis': error_analysis.get('semantic_analysis', {})
                })

        return failed_tests

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description='Match failed tests with JIRA tickets using optimized enhanced relevance scoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
# Default: Optimized scoring with 50%% threshold, top 5 results
python script.py

# Show score breakdown
python script.py --show-score-breakdown

# Lower threshold to see more matches
python script.py --threshold 0.30

# Very strict matching (70%%+)
python script.py --threshold 0.70 --show-score-breakdown
"""
    )

    parser.add_argument('--threshold', type=float, default=0.50,
                        help='Minimum similarity threshold (0.0-1.0). Default: 0.50 (50%%)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top matches to show per test. Use 0 for all. Default: 5')
    parser.add_argument('--failed-tests', type=str,
                        default='failed_tests.txt',
                        help='Path to failed tests file')
    parser.add_argument('--jira-tickets', type=str,
                        default='activities/jira_issues.json',
                        help='Path to JIRA tickets JSON file')
    parser.add_argument('--output', type=str, default='matching_results.json',
                        help='Output JSON file path. Default: matching_results.json')
    parser.add_argument('--show-details', action='store_true', default=True,
                        help='Show detailed match scores')
    parser.add_argument('--show-score-breakdown', action='store_true', default=False,
                        help='Show detailed score breakdown including bonuses')
    parser.add_argument('--include-comments', action='store_true', default=True,
                        help='Include JIRA comments in matching (default: True)')
    parser.add_argument('--no-comments', action='store_true', default=False,
                        help='Exclude JIRA comments from matching')
    parser.add_argument('--max-comments', type=int, default=10,
                        help='Maximum number of recent comments to include. Default: 10')

    args = parser.parse_args()

    include_comments = args.include_comments and not args.no_comments

    if not 0.0 <= args.threshold <= 1.0:
        print("❌ Error: Threshold must be between 0.0 and 1.0")
        return

    top_n = None if args.top_n == 0 else args.top_n

    print("=" * 80)
    print("OPTIMIZED SEMANTIC ERROR ANALYSIS & JIRA MATCHING")
    print("=" * 80)
    print(f"⚙️  Configuration:")
    print(f"   - Minimum similarity threshold: {args.threshold:.0%}")
    print(f"   - Top matches per test: {'All' if top_n is None else top_n}")
    print(f"   - Include comments: {'Yes' if include_comments else 'No'}")
    if include_comments:
        print(f"   - Max comments per ticket: {args.max_comments}")
    print(f"   - Optimized scoring: Enabled (Pre-computed embeddings)")
    print(f"   - Show score breakdown: {'Yes' if args.show_score_breakdown else 'No'}")
    print("=" * 80)

    print("\n📖 Loading failed tests with semantic analysis...")
    failed_tests = read_failed_tests_new_format(args.failed_tests)

    if not failed_tests:
        print("❌ No failed tests found or error reading the file.")
        return

    print(f"\n✅ Found {len(failed_tests)} failed tests")

    print("\n" + "=" * 80)
    print("SEMANTIC ANALYSIS RESULTS")
    print("=" * 80)
    for i, test in enumerate(failed_tests, 1):
        print(f"\n{i}. {test.get('test_path', 'Unknown')}")
        print(f"   📝 Description: {test.get('test_description')}")
        print(f"   🔍 Test Case ID: {test.get('test_case_id')}")
        print(f"   ❌ Error: {test.get('error')}")
        print(f"   📊 Category: {test.get('category')} | Root Cause: {test.get('root_cause')}")
        print(f"   ✓ Confidence: {test.get('confidence')}")

    print("\n\n🎫 Loading Jira tickets...")
    try:
        with open(args.jira_tickets, 'r') as f:
            jira_tickets = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: JIRA tickets file not found: {args.jira_tickets}")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in JIRA tickets file")
        return

    print(f"✅ Found {len(jira_tickets)} Jira tickets")

    print("\n🤖 Initializing optimized spaCy matcher...")
    matcher = OptimizedJiraTicketMatcher(model_name="en_core_web_md")

    results = matcher.find_matching_tickets(
        failed_tests=failed_tests,
        jira_tickets=jira_tickets,
        top_n=top_n,
        threshold=args.threshold,
        use_multi_field=True,
        include_comments=include_comments,
        max_comments=args.max_comments
    )

    # Calculate statistics
    total_matches = sum(len(r['matches']) for r in results)
    tests_with_matches = len([r for r in results if r['matches']])
    tests_without_matches = len(results) - tests_with_matches

    if total_matches > 0:
        avg_similarity = sum(
            match['similarity']
            for r in results
            for match in r['matches']
        ) / total_matches

        avg_exact_bonus = sum(
            match.get('exact_match_bonus', 0)
            for r in results
            for match in r['matches']
        ) / total_matches

        avg_status_bonus = sum(
            match.get('status_bonus', 0)
            for r in results
            for match in r['matches']
        ) / total_matches
    else:
        avg_similarity = 0
        avg_exact_bonus = 0
        avg_status_bonus = 0

    print(f"\n📊 Match Statistics:")
    print(f"   - Tests analyzed: {len(results)}")
    print(f"   - Tests with matches ≥{args.threshold:.0%}: {tests_with_matches}")
    print(f"   - Tests without matches: {tests_without_matches}")
    print(f"   - Total matches found: {total_matches}")
    if tests_with_matches > 0:
        print(
            f"   - Average matches per test (with matches): {total_matches / tests_with_matches:.1f}")
    if total_matches > 0:
        print(f"   - Average similarity score: {avg_similarity:.1%}")
        print(f"   - Average exact match bonus: {avg_exact_bonus:.1%}")
        print(f"   - Average status bonus: {avg_status_bonus:+.1%}")

    # Print results
    matcher.print_results(results, show_details=args.show_details,
                          show_score_breakdown=args.show_score_breakdown)

    # Save to JSON
    print(f"\n💾 Saving results to {args.output}...")
    try:
        with open(args.output, 'w') as f:
            clean_results = []
            for r in results:
                clean_r = r.copy()
                clean_r['matches'] = [
                    {k: v for k, v in m.items() if k != 'ticket'}
                    for m in r['matches']
                ]
                clean_results.append(clean_r)

            output_data = {
                'metadata': {
                    'threshold': args.threshold,
                    'top_n': args.top_n,
                    'include_comments': include_comments,
                    'max_comments': args.max_comments if include_comments else 0,
                    'enhanced_scoring': True,
                    'optimized': True,
                    'total_tests': len(results),
                    'tests_with_matches': tests_with_matches,
                    'total_matches': total_matches,
                    'average_similarity': avg_similarity if total_matches > 0 else 0,
                    'average_exact_match_bonus': avg_exact_bonus if total_matches > 0 else 0,
                    'average_status_bonus': avg_status_bonus if total_matches > 0 else 0
                },
                'results': clean_results
            }

            json.dump(output_data, f, indent=2)
        print(f"✅ Results saved to '{args.output}'")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()