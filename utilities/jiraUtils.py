import time
import json
from jira import JIRA
import spacy
import re
from datetime import datetime, timedelta
from nltk.stem import PorterStemmer
import nltk
import os
from typing import Optional, List

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class JiraIssueFilterAgent:
    def __init__(self, jira_url, username, api):
        """Initialize JIRA connection, Spacy NLP model, and NLTK stemmer."""
        try:
            self.jira = JIRA(server=jira_url, basic_auth=(username, api))
            self.jira_connected = True
            print("Connected to JIRA successfully!")
        except Exception as e:
            print(f"Failed to connect to JIRA: {e}")

        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp_loaded = True
            print("Loaded spaCy model successfully!")
        except Exception as e:
            print(f"Failed to load spaCy model: {e}")
            print("Make sure to install it with: python -m spacy download en_core_web_sm")

        self.stemmer = PorterStemmer()
        if not (self.jira_connected and self.nlp_loaded):
            print("Warning: Some components failed to initialize")

    def fetch_issues(self, jql_query):
        """Fetch issues from JIRA using JQL query."""
        try:
            # issues = self.jira.search_issues(jql_query, maxResults=100)
            issues = self.jira.search_issues(jql_query, maxResults=0)
            print(f"Fetched {len(issues)} issues from JIRA.")
            return issues
        except Exception as e:
            print(f"Error fetching issues: {e}")
            return []

    def parse_prompt(self, prompt):
        """Parse user prompt using Spacy to extract filter criteria and keywords."""
        prompt_lower = prompt.lower()
        doc = self.nlp(prompt_lower)

        filters = {
            "status": None,
            "priority": None,
            "assignee": None,
            "due_date": None,
            "label": None,
            "keywords": [],
            "phrase": None,
            "negated_terms": []
        }
        # Define synonyms for statuses and priorities
        status_map = {
            "open": ["open", "to do", "todo", "not started"],
            "in progress": ["in progress", "ongoing", "started"],
            "closed": ["closed", "done", "resolved", "completed", "fixed"]
        }
        priority_map = {
            "high": ["high", "urgent", "critical", "important"],
            "medium": ["medium", "normal", "moderate"],
            "low": ["low", "minor", "trivial"]
        }
        # Extended stop words list including common verbs
        stop_words = {
            "issues", "tickets", "jira", "the", "a", "an", "issue", "ticket",
            "to", "with", "about", "show", "find", "get", "related",
            "is", "are", "was", "were", "be", "being", "been", "am",
            "has", "have", "had", "do", "does", "did", "can", "could",
            "will", "would", "shall", "should", "may", "might", "must",
            "and", "or", "but", "if", "then", "else", "when", "where", "why",
            "how", "what", "which", "who", "whom", "whose", "this", "that",
            "these", "those", "for", "of", "by", "at", "in", "on", "upon"
        }
        # Basic negation terms
        negation_terms = ["not", "isn't", "doesn't", "don't", "no", "cannot", "can't"]
        # Process negation phrases to extract negated terms
        negation_phrases = [
            (r'is\s+not\s+(\w+ing)', 1),  # "is not working" -> "working"
            (r'does\s+not\s+(\w+)', 1),  # "does not work" -> "work"
            (r'not\s+(\w+ing)', 1),  # "not working" -> "working"
            (r'no\s+(\w+)', 1),  # "no response" -> "response"
            (r'isn\'t\s+(\w+ing)', 1),  # "isn't working" -> "working"
            (r'doesn\'t\s+(\w+)', 1),  # "doesn't work" -> "work"
            (r'don\'t\s+(\w+)', 1),  # "don't work" -> "work"
            (r'cannot\s+(\w+)', 1),  # "cannot connect" -> "connect"
            (r'can\'t\s+(\w+)', 1)  # "can't connect" -> "connect"
        ]
        negated_terms = []
        for pattern, group_idx in negation_phrases:
            matches = re.search(pattern, prompt_lower)
            if matches:
                negated_term = matches.group(group_idx)
                if negated_term not in negated_terms:
                    negated_terms.append(negated_term)
        filters["negated_terms"] = negated_terms

        # Extract status with improved context detection
        for status, synonyms in status_map.items():
            for synonym in synonyms:
                # Match status in more specific contexts
                if (re.search(r'\bstatus\s+(?:is|are|=)\s+' + re.escape(synonym) + r'\b', prompt_lower) or
                        re.search(r'\b' + re.escape(synonym) + r'\s+(?:issues?|tickets?)\b', prompt_lower)):
                    filters["status"] = status
                    break
            if filters["status"]:
                break

        # Extract priority
        for priority, synonyms in priority_map.items():
            for synonym in synonyms:
                # Match priority in more specific contexts
                if (re.search(r'\bpriority\s+(?:is|=)\s+' + re.escape(synonym) + r'\b', prompt_lower) or
                        re.search(r'\b' + re.escape(synonym) + r'\s+priority\b', prompt_lower) or
                        re.search(r'\b' + re.escape(synonym) + r'\s+(?:issues?|tickets?)\b', prompt_lower)):
                    filters["priority"] = priority
                    break
            if filters["priority"]:
                break

        # Extract assignee
        assignee_patterns = [
            r'assigned\s+to\s+(\w+)',
            r'(\w+)\'s\s+issues',
            r'(\w+)\s+is\s+working\s+on',
        ]

        for pattern in assignee_patterns:
            matches = re.search(pattern, prompt_lower)
            if matches:
                filters["assignee"] = matches.group(1)
                break

        # Extract due date
        if "due" in prompt_lower:
            if "this week" in prompt_lower:
                filters["due_date"] = (datetime.now(), datetime.now() + timedelta(days=7))
            elif "today" in prompt_lower:
                filters["due_date"] = (datetime.now(), datetime.now() + timedelta(days=1))
            elif "tomorrow" in prompt_lower:
                tomorrow = datetime.now() + timedelta(days=1)
                filters["due_date"] = (tomorrow, tomorrow + timedelta(days=1))
            elif "overdue" in prompt_lower:
                filters["due_date"] = (None, datetime.now())

        # Extract labels
        label_patterns = [
            r'label(?:ed)?\s+(?:with|as)\s+(\w+)',
            r'with\s+(\w+)\s+label',
            r'tag(?:ged)?\s+(?:with|as)\s+(\w+)',
        ]

        for pattern in label_patterns:
            matches = re.search(pattern, prompt_lower)
            if matches:
                filters["label"] = matches.group(1)
                break

        # Extract keywords - skip common verbs and stop words
        tokens = [token.text for token in doc if not token.is_punct]
        keywords = []
        for token in tokens:
            # Skip if it's a stop word, negation term, or already captured as a negated term
            if (token in stop_words or
                    token in negation_terms or
                    token in negated_terms or
                    any(token in synonyms for status_terms in status_map.values() for synonyms in status_terms) or
                    any(token in synonyms for priority_terms in priority_map.values() for synonyms in priority_terms)):
                continue
            keywords.append(token)
        # Remove duplicates and short terms (likely not meaningful)
        keywords = [k for k in keywords if len(k) > 2]
        keywords = list(dict.fromkeys(keywords))  # Remove duplicates while preserving order
        if keywords:
            filters["keywords"] = keywords
            filters["phrase"] = " ".join(keywords)  # Store the main search phrase
        # print(f"Parsed filters: {filters}")  # Debug print to see extracted filters
        return filters

    def _is_word_match(self, keyword, text):
        """Check if keyword appears as a word in the text with stemming."""
        if not text:
            return False
        # Stem the keyword and text for more flexible matching
        stemmed_keyword = self.stemmer.stem(keyword.lower())
        # Process each word in text separately for stemming
        stemmed_text_words = [self.stemmer.stem(word) for word in text.lower().split()]
        stemmed_text = " ".join(stemmed_text_words)
        # Check for whole word match using regex with word boundaries
        pattern = r'\b' + re.escape(stemmed_keyword) + r'\b'
        return bool(re.search(pattern, stemmed_text))

    def _is_phrase_match(self, phrase, text):
        """Check if phrase appears in the text with flexible matching."""
        if not text or not phrase:
            return False
        # For multi-word phrases, try exact matching first
        phrase_lower = phrase.lower()
        text_lower = text.lower()
        # Direct substring match (less strict)
        if phrase_lower in text_lower:
            return True
        # Stem the phrase and words in text for more flexible matching
        stemmed_phrase_words = [self.stemmer.stem(word) for word in phrase_lower.split()]
        stemmed_phrase = " ".join(stemmed_phrase_words)
        stemmed_text_words = [self.stemmer.stem(word) for word in text_lower.split()]
        stemmed_text = " ".join(stemmed_text_words)
        # Check for stemmed phrase match
        return stemmed_phrase in stemmed_text

    def filter_issues(self, issues, **filters):
        """Filter issues based on criteria with improved matching and scoring."""
        matching_issues = []
        # Extract filters
        status = filters.get("status")
        priority = filters.get("priority")
        assignee = filters.get("assignee")
        due_date = filters.get("due_date")
        label = filters.get("label")
        keywords = filters.get("keywords", [])
        phrase = filters.get("phrase")
        negated_terms = filters.get("negated_terms", [])
        # Add these filtering parameters
        min_relevance = filters.get("min_relevance", 0.40)  # Default 40%
        require_summary_match = filters.get("require_summary_match",
                                            False)  # Default to searching both title and description
        for issue in issues:
            matches = True      # Basic criteria check
            if status and issue.fields.status.name.lower() != status:       # Check status (case-insensitive)
                matches = False
            if priority and issue.fields.priority.name.lower() != priority:     # Check priority (case-insensitive)
                matches = False
            if assignee and (not issue.fields.assignee or           # Check assignee
                             assignee.lower() not in issue.fields.assignee.displayName.lower()):
                matches = False
            # Check due date
            if due_date:
                issue_due = getattr(issue.fields, 'duedate', None)
                if issue_due:
                    due_date_obj = datetime.strptime(issue_due, "%Y-%m-%d")
                    start, end = due_date
                    if start and due_date_obj < start:
                        matches = False
                    if end and due_date_obj > end:
                        matches = False
                else:
                    matches = False
            # Check label
            if label and (not hasattr(issue.fields, 'labels') or
                          not any(label.lower() in lbl.lower() for lbl in issue.fields.labels)):
                matches = False
            # Skip text search if basic filters already rule out this issue
            if not matches:
                continue
            # Get issue content
            summary = issue.fields.summary if issue.fields.summary else ""
            description = issue.fields.description if issue.fields.description else ""
            # Initialize scoring variables
            relevance_score = 0
            match_info = []
            # Check if there's a summary match when required
            has_summary_match = False
            if require_summary_match:
                if phrase and self._is_phrase_match(phrase, summary):
                    has_summary_match = True
                else:
                    for keyword in keywords:
                        if self._is_word_match(keyword, summary):
                            has_summary_match = True
                            break

                if not has_summary_match:
                    continue  # Skip this issue since it only matches in description
            # Check for negative terms - exclude issues with these terms
            has_negated_term = False
            for neg_term in negated_terms:
                if self._is_word_match(neg_term, summary) or self._is_word_match(neg_term, description):
                    has_negated_term = True
                    match_info.append(f"Excluded: contains negated term '{neg_term}'")
                    break
            if has_negated_term:
                continue  # Skip this issue if it contains a negated term
            # Text search logic - improved phrase and keyword matching
            if phrase:
                # Check for phrase match in summary (highest score)
                if self._is_phrase_match(phrase, summary):
                    phrase_score = 10 * len(phrase.split())  # Longer phrases get higher scores
                    relevance_score += phrase_score
                    match_info.append(f"Phrase '{phrase}' found in summary")

                # Check for phrase match in description
                if description and self._is_phrase_match(phrase, description):
                    phrase_score = 8 * len(phrase.split())
                    relevance_score += phrase_score
                    match_info.append(f"Phrase '{phrase}' found in description")

                # If no phrase match or low score, check for individual keywords
                if keywords and relevance_score < 10:
                    for keyword in keywords:
                        if self._is_word_match(keyword, summary):
                            relevance_score += 5
                            match_info.append(f"Keyword '{keyword}' found in summary")
                        if description and self._is_word_match(keyword, description):
                            relevance_score += 3
                            match_info.append(f"Keyword '{keyword}' found in description")

            elif keywords:
                # If no phrase but we have keywords, check each keyword
                for keyword in keywords:
                    if self._is_word_match(keyword, summary):
                        relevance_score += 5
                        match_info.append(f"Keyword '{keyword}' found in summary")
                    if description and self._is_word_match(keyword, description):
                        relevance_score += 3
                        match_info.append(f"Keyword '{keyword}' found in description")

            # Only include issues with a significant relevance score
            if relevance_score > 2:  # Basic scoring threshold
                # Calculate normalized score (0-1 range)
                max_possible_score = 20  # Adjust based on your scoring system
                normalized_score = min(relevance_score / max_possible_score, 1.0)

                # Check against minimum relevance percentage
                if normalized_score >= min_relevance:
                    # Add the issue to results
                    matching_issues.append((issue, normalized_score, match_info))

        # Sort issues by relevance score (highest first)
        matching_issues.sort(key=lambda x: x[1], reverse=True)
        return matching_issues

    def display_issues(self, matching_issues):
        """Display filtered issues with their match information and relevance scores."""
        if not matching_issues:
            print("No issues found matching the criteria.")
            return False
        print("\n" + "=" * 80)
        print(f"Found {len(matching_issues)} matching issues:")
        print("=" * 80)

        for idx, (issue, score, match_info) in enumerate(matching_issues, 1):
            # Format score as percentage
            percentage_score = f"{score * 100:.1f}%"
            print(f"\n{idx}. Issue Key: {issue.key}")
            print(f"   Summary: {issue.fields.summary}")
            print(f"   Status: {issue.fields.status.name}")
            print(f"   Priority: {issue.fields.priority.name}")
            assignee = getattr(issue.fields.assignee, 'displayName',
                               'Unassigned') if issue.fields.assignee else 'Unassigned'
            print(f"   Assignee: {assignee}")
            print(f"   Relevance: {percentage_score}")
            if match_info:
                print("   Match Details:")
                for info in match_info:
                    print(f"     - {info}")
            print("-" * 80)
        return True

    def find_board_id_for_project(self,jira: JIRA, project_key: str) -> Optional[int]:
        start_at = 0
        max_results = 50
        while True:
            boards = jira.boards(startAt=start_at, maxResults=max_results)
            if not boards:
                break
            for b in boards:
                proj_key = getattr(getattr(b, "location", None), "projectKey", None)
                if proj_key == project_key:
                    return b.id
            start_at += max_results
        return None

    def list_active_sprints(self,jira: JIRA, board_id: int) -> List:
        start_at = 0
        max_results = 50
        sprints = []
        while True:
            batch = jira.sprints(board_id, state="active", startAt=start_at, maxResults=max_results)
            if not batch:
                break
            sprints.extend(batch)
            if len(batch) < max_results:
                break
            start_at += max_results
        return sprints

    def create_jira_ticket(self, failed_test):
        """Create a new JIRA ticket for a failed test with no matching issues and attach the test report."""
        try:
            # Extract data from the failed test
            test_name = failed_test['test_path'].split("::")[-1]  # Extract just the test name from the path

            summary = f"Test Failure: {test_name}"
            description = (
                f"Failed Test: {failed_test['test_path']}\n\n"
                f"Error: {failed_test['error']}"
            )

            # Add key terms if available
            if 'key_terms' in failed_test and failed_test['key_terms']:
                description += f"\n\nKey Terms: {', '.join(failed_test['key_terms'])}"

            # Define the issue fields
            issue_dict = {
                'project': {'key': project},
                'summary': summary,
                'description': description,
                'issuetype': {'name': 'Bug'},
                # 'priority': {'name': 'Medium'}  # Adjust based on your needs
            }

            # Create the issue
            new_issue = self.jira.create_issue(fields=issue_dict)
            print(f"New issue created: {new_issue.key}")

            # Attach the latest test report ZIP file to the ticket
            reports_dir = "Reports"
            # Find the most recent ZIP file in the Reports directory
            zip_files = [f for f in os.listdir(reports_dir) if f.endswith('.zip') and f.startswith('test_report_')]

            if zip_files:
                # Sort by creation time (newest first)
                zip_files.sort(key=lambda x: os.path.getctime(os.path.join(reports_dir, x)), reverse=True)
                latest_zip = os.path.join(reports_dir, zip_files[0])

                # Attach the file to the JIRA issue
                with open(latest_zip, 'rb') as attachment:
                    self.jira.add_attachment(issue=new_issue.key, attachment=attachment,
                                             filename=zip_files[0])
                print(f"Attached test report {zip_files[0]} to JIRA ticket {new_issue.key}")
            else:
                print("No test report ZIP file found to attach to JIRA ticket")

            return new_issue
        except Exception as e:
            print(f"Error creating JIRA ticket: {e}")
            return None

def extract_key_terms_dynamic(test_path, error_text):
    """Dynamically extract key terms from test failure information using NLP techniques."""
    key_terms = []
    # 1. Extract the test name for context
    test_name_parts = test_path.split('::')[-1].replace('test_', '').replace('_', ' ').split()
    # 2. Clean the error text
    cleaned_error = re.sub(r'E\s+', '', error_text).strip()
    # 3. Extract domain-specific elements
    # URLs
    urls = re.findall(r'(https?://[^\s]+)', cleaned_error)
    for url in urls:
        domain = url.split('/')[2]
        key_terms.append(f"URL issue with {domain}")
    # Email addresses
    emails = re.findall(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', cleaned_error)
    if emails:
        key_terms.append("credential issue")
    # 4. Identify common error patterns
    error_patterns = {
        'assert.*==': 'equality check failed',
        'assert.*!=': 'inequality check failed',
        'assert.*in': 'membership check failed',
        'timeout': 'timeout error',
        'not found': 'element not found',
        'permission': 'permission issue',
        'denied': 'access denied',
        'unauthorized': 'authorization issue',
        'login': 'authentication problem',
        'password': 'credential failure',
        'url': 'navigation problem',
        'redirect': 'redirection issue',
        'expected.*but got': 'unexpected result',
        'failed': 'test failure',
        'error': 'error occurred',
        'exception': 'exception raised'
    }
    for pattern, term in error_patterns.items():
        if re.search(pattern, cleaned_error.lower()):
            key_terms.append(term)

    # 5. Extract key nouns and verbs using basic NLP
    # Remove common assertion boilerplate
    cleaned_text = re.sub(r'AssertionError:', '', cleaned_error)
    # Get words, excluding very common ones
    words = [w.lower() for w in re.findall(r'\b\w+\b', cleaned_text)
             if len(w) > 3 and w.lower() not in {
                 'with', 'from', 'that', 'this', 'have', 'were', 'assertion',
                 'assert', 'error', 'the', 'and', 'for', 'not'
             }]
    # Add unique important words
    key_terms.extend([w for w in words if w not in [kt.lower() for kt in key_terms]])
    # 6. Add test context based on the test name
    test_context = ' '.join(test_name_parts)
    if test_context and test_context.lower() not in [kt.lower() for kt in key_terms]:
        key_terms.append(f"{test_context} issue")
    # 7. Remove duplicates while preserving order
    unique_terms = []
    for term in key_terms:
        if term not in unique_terms:
            unique_terms.append(term)
    return unique_terms[:5]  # Limit to top 5 most relevant terms

def read_failed_tests(filepath):
    """Read and parse the failed tests summary file."""
    failed_tests = []
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        # Extract individual test failures
        test_sections = content.split('Test: ')[1:]
        for section in test_sections:
            lines = section.split('\n')
            test_path = lines[0].strip()
            # Extract the error message more cleanly
            error_text = ""
            for line in lines:
                if line.strip().startswith('Error:'):
                    error_text = line.replace('Error:', '').strip()
                    break
            # Extract test name from path
            test_name = test_path.split('::')[-1]
            # Extract class name if available
            class_name = ""
            if '::' in test_path:
                parts = test_path.split('::')
                if len(parts) > 1:
                    class_name = parts[1]

            # Dynamically extract key terms
            # key_terms = extract_key_terms_dynamic(test_path, error_text)
            failed_tests.append({
                'test_path': test_path,
                # 'test_name': test_name,
                # 'class_name': class_name,
                'error': error_text,
                # 'key_terms': key_terms
            })
        return failed_tests
    except Exception as e:
        print(f"Error reading failed tests file: {e}")
        return []

def main():
    # Initialize the agent
    agent = JiraIssueFilterAgent(jira_url, username, api)
    # Fetch issues from the project
    # jql = f"project = {project} AND type = Bug AND created >= '2025-09-01' AND created <= '2025-10-27' ORDER BY created DESC"
    # jql ='project = ASPEN AND type = Bug AND created >= "2025-09-01" AND created <= "2025-10-27" ORDER BY created DESC'
    # jql = "project = ASPEN AND type = Bug AND reporter = 712020:70d5fd5b-8fe1-448a-a156-ed73db40ff9f ORDER BY created DESC"
    # jql = 'project = ASPEN AND type = Bug AND reporter IN (712020:19df65b3-8cb0-4775-ad8a-db64550fb854, 712020:70d5fd5b-8fe1-448a-a156-ed73db40ff9f)'
    jql = 'project IN (ASPEN, LT) AND type = Bug AND reporter IN (712020:19df65b3-8cb0-4775-ad8a-db64550fb854, 712020:70d5fd5b-8fe1-448a-a156-ed73db40ff9f, 712020:78f31852-aa02-447d-a128-ee44eefb02d2, 712020:7e5f32dd-6e46-44ed-883d-6609920c0269) ORDER BY created DESC'
    issues = agent.fetch_issues(jql)


    ###################    REMOVE THIS CODE
    # open a new file in write mode
    with open("activities/jira_issues.json", "w") as file:
        json.dump([issue.raw for issue in issues], file, indent=4)

    print("Text written to output.txt successfully!")
    time.sleep(7)


    # Set default filtering options
    min_relevance = 0.90  # 40%
    # min_relevance = 0.40  # 40%
    require_summary_match = False  # Search in both title and description
    print(f"\nActive settings: {min_relevance * 100:.0f}% minimum relevance, "
          f"searching in both title and description")
    tests_without_matches = []
    # Read failed tests from file
    failed_tests = read_failed_tests('failed_tests.txt')
    if not failed_tests:
        print("No failed tests found or error reading the file.")
        return
    print(f"Found {len(failed_tests)} failed tests. Processing each one...")

    for idx, test in enumerate(failed_tests, 1):
        print(f"\n==== Processing Failed Test #{idx} ====")
        print(f"Test: {test['test_path']}")
        print(f"Error: {test['error']}")
        # Generate prompt from test failure
        # Extract meaningful keywords from test paths and error mes
        # sages
        test_name = test['test_path'].split('::')[-1]  # Get the test method name

        # Create a prompt based on test name and error
        prompt = f"Find issues related to {test_name} failure with {test['error']}"
        # print(f"Generated prompt: '{prompt}'")

        # Parse prompt to extract filters
        filters = agent.parse_prompt(prompt)

        # Add the filtering parameters
        filters["min_relevance"] = min_relevance
        filters["require_summary_match"] = require_summary_match

        # Filter issues based on parsed criteria
        filtered_issues = agent.filter_issues(issues, **filters)

        # Display results for this test
        print(f"\nResults for failed test: {test['test_path']}")
        has_matches = agent.display_issues(filtered_issues)
        if not has_matches:
            tests_without_matches.append(test)
        print("\n" + "=" * 80)


    # After processing all tests, prompt user about creating tickets
    if tests_without_matches:
        prompt = input(
            "\nDo you want to create JIRA ticket for unfounded Issues? (Press 'y' or 'yes' to accept) or (or 'exit' to quit): ")

        if prompt.lower() in ['y', 'yes']:
            print("Hurray")

            # Ask about each test individually
            for test in tests_without_matches:
                test_path = test['test_path']
                test_error = test['error']

                confirm = input(f"\nCreate JIRA ticket for: \nTest: {test_path}\nError: {test_error}\n(y/n): ")

                if confirm.lower() in ['y', 'yes']:
                    # Call your create_jira_ticket function
                    new_issue = agent.create_jira_ticket(test)

                    if new_issue:
                        print(f"Created new JIRA ticket: {new_issue.key}")
                    else:
                        print("Failed to create JIRA ticket.")
                else:
                    print(f"Skipping ticket creation for {test_path}")

            print("JIRA issue matching completed successfully.")
        elif prompt.lower() == 'exit':
            print("Exiting program.")
            return
        else:
            print("No JIRA tickets created.")

    print("Process complete!")


if __name__ == "__main__":
    main()



