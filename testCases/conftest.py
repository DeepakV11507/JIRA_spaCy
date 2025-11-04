import pytest
from selenium import webdriver
from utilities.readProperties import load_yaml_file
import zipfile
import os
import os
import sys
import zipfile
import subprocess
from datetime import datetime

from datetime import datetime
import subprocess
import os
import sys
import shutil
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from utilities.readProperties import load_yaml_file
import zipfile
import os
import time
import sys
import subprocess
from datetime import datetime, timedelta
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.edge.options import Options as EdgeOptions


@pytest.fixture()
def setup(browser):
    print("setup")
    if browser == "chrome":
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--start-maximized")  # This avoids the separate maximize call
        # Optional: Add these for more stability
        # chrome_options.add_argument("--no-sandbox")
        # chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=chrome_options)
    elif browser == "firefox":
        firefox_options = FirefoxOptions()
        firefox_options.add_argument("--headless")  # Enable headless mode
        driver = webdriver.Firefox(options=firefox_options)
        driver.maximize_window()  # Firefox usually works fine with this method
    elif browser == "edge":
        edge_options = EdgeOptions()
        driver = webdriver.Edge(options=edge_options)
        driver.maximize_window()  # Edge usually works fine with this method
    else:
        # Default to Chrome if no browser specified
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--start-maximized")
        driver = webdriver.Chrome(options=chrome_options)

    yield driver

    driver.quit()


def pytest_addoption(parser):
    parser.addoption("--browser")


@pytest.fixture()
def browser(request):
    return request.config.getoption("--browser")


@pytest.hookimpl(optionalhook=True)
def pytest_metadata(metadata):
    metadata.pop("JAVA_HOME", None)
    metadata.pop("Plugins", None)


# FOR YAML ONLY
@pytest.fixture(scope="session")
def config():
    return load_yaml_file("Configurations/config.yaml")


@pytest.fixture(scope="module")
def load_common_info(config):
    return {
        "baseurl": config["baseURL"],
        "email": config["email"],
        "password": config["password"]
    }


def pytest_configure(config):
    """
    Called after command line options have been parsed and
    before any test is executed. This is the perfect place
    to clean the Reports folder.
    """
    # Define the reports directory path
    reports_dir = "Reports"

    # Check if the directory exists
    if os.path.exists(reports_dir):
        print(f"\n=== Cleaning Reports folder ===")

        try:
            # Option 1: Remove all files within the directory but keep the directory
            for item in os.listdir(reports_dir):
                item_path = os.path.join(reports_dir, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

            print(f"Successfully cleaned {reports_dir} folder")
        except Exception as e:
            print(f"Error cleaning {reports_dir} folder: {e}")
    else:
        # Directory doesn't exist, no need to clean it
        print(f"\n=== Reports folder does not exist yet ===")

def pytest_sessionfinish(session, exitstatus):
    """
    Called after the test run finishes. Writes only test case names and
    assertion error messages to a text file, then runs the JIRA utility.
    """
    if exitstatus != 0:  # Non-zero exit status means some tests failed
        with open("failed_tests.txt", "w") as f:
            f.write("=== FAILED TESTS SUMMARY ===\n\n")

            failed_count = 0
            reporter = session.config.pluginmanager.get_plugin('terminalreporter')
            if reporter:
                for test_id, reports in reporter.stats.items():
                    if test_id == 'failed':
                        for rep in reports:
                            failed_count += 1
                            # Get just the test name
                            test_name = rep.nodeid
                            f.write(f"Test: {test_name}\n")

                            # Extract only the AssertionError message
                            if hasattr(rep, "longrepr"):
                                error_text = str(rep.longrepr)
                                # Find the AssertionError line
                                for line in error_text.split('\n'):
                                    if 'AssertionError:' in line:
                                        # Extract just the message part
                                        error_message = line.strip()
                                        f.write(f"Error: {error_message}\n")
                                        break

                            f.write("-" * 80 + "\n\n")

            f.write(f"Total Failed Tests: {failed_count}\n")

        failed_count = 3
        src = "activities/output.txt"  # source file
        dst = "failed_tests.txt"  # destination file

        # Empty the destination file first
        open(dst, "w").close()

        # Copy content from source to destination
        with open(src, "r", encoding="utf-8") as f_src, open(dst, "a", encoding="utf-8") as f_dst:
            for line in f_src:
                f_dst.write(line)






        # If there are failed tests, run the JIRA utility
        if failed_count > 0:
            # Create timestamp for unique report naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create the report directory structure
            reports_dir = "Reports"
            assets_dir = os.path.join(reports_dir, "assets")

            # Create directories if they don't exist
            os.makedirs(reports_dir, exist_ok=True)
            os.makedirs(assets_dir, exist_ok=True)

            # Path for the HTML report and CSS file
            html_report_path = os.path.join(reports_dir, "report.html")
            css_file_path = os.path.join(assets_dir, "style.css")

            # Calculate test statistics
            total = len(reporter.stats.get("passed", [])) + len(reporter.stats.get("failed", [])) + len(
                reporter.stats.get("skipped", [])) + len(reporter.stats.get("error", []))
            passed = len(reporter.stats.get("passed", []))
            failed = len(reporter.stats.get("failed", []))
            error = len(reporter.stats.get("error", []))
            skipped = len(reporter.stats.get("skipped", []))

            # Create CSS content with the enhanced styling
            css_content = """
body {
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f8f9fa;
    color: #212529;
    line-height: 1.5;
}

h1, h2, h3 {
    font-family: 'Montserrat', 'Segoe UI', sans-serif;
    color: #2c3e50;
}

h1 {
    text-align: center;
    padding: 15px;
    background: linear-gradient(135deg, #4CAF50, #2E7D32);
    color: white;
    border-radius: 8px;
    margin-top: 10px;
    box-shadow: 0 3px 6px rgba(0,0,0,0.16);
    font-weight: 600;
}

.report-timestamp {
    text-align: center;
    color: #6c757d;
    margin-bottom: 30px;
    font-style: italic;
}

.summary-container {
    padding: 20px;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 30px;
}

.results-container {
    padding: 20px;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-bottom: 20px;
    font-size: 14px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-radius: 8px;
    overflow: hidden;
}

th, td {
    padding: 12px 15px;
    text-align: left;
}

th {
    background-color: #343a40;
    color: white;
    font-weight: 600;
    position: sticky;
    top: 0;
}

tr:nth-child(even) {
    background-color: #f2f2f2;
}

td {
    border-bottom: 1px solid #e9ecef;
    border-right: 1px solid #e9ecef;
}

td:last-child {
    border-right: none;
}

tr:last-child td {
    border-bottom: none;
}

tr:hover {
    background-color: #eff6ff;
}

.passed {
    background-color: #e8f5e9;
}

.passed:hover {
    background-color: #c8e6c9;
}

.failed {
    background-color: #ffebee;
}

.failed:hover {
    background-color: #ffcdd2;
}

.skipped {
    background-color: #fff8e1;
}

.skipped:hover {
    background-color: #ffecb3;
}

.error {
    background-color: #ffebee;
}

.error:hover {
    background-color: #ffcdd2;
}

.summary-box {
    display: inline-block;
    width: 22%;
    margin: 0 1%;
    padding: 15px;
    text-align: center;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}

.summary-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.total-box {
    background: linear-gradient(135deg, #b0bec5, #78909c);
    color: white;
    border: none;
}

.pass-box {
    background: linear-gradient(135deg, #81c784, #4caf50);
    color: white;
    border: none;
}

.fail-box {
    background: linear-gradient(135deg, #ef5350, #e53935);
    color: white;
    border: none;
}

.skip-box {
    background: linear-gradient(135deg, #fff176, #ffd54f);
    color: #5d4037;
    border: none;
}

.summary-value {
    font-size: 28px;
    font-weight: 700;
    margin: 5px 0;
}

.summary-label {
    font-size: 14px;
    font-weight: 500;
}

.error-message {
    font-family: 'Roboto Mono', 'Consolas', monospace;
    white-space: pre-wrap;
    color: #d32f2f;
    font-size: 13px;
}

.debug-info {
    font-family: 'Roboto Mono', 'Consolas', 'Monaco', monospace;
    font-size: 13px;
    line-height: 1.6;
    background-color: #1a1a1a;
    color: #f8f8f2;
    border-radius: 6px;
    padding: 15px;
    margin-top: 10px;
    white-space: pre-wrap;
    overflow-x: auto;
    max-height: 0px;
    opacity: 0;
    transition: max-height 0.5s ease-in-out, opacity 0.4s ease-in-out;
    word-wrap: break-word;
    box-shadow: 0 3px 6px rgba(0,0,0,0.16);
}

.debug-toggle {
    cursor: pointer;
    background-color: #3f51b5;
    color: white;
    padding: 6px 12px;
    border-radius: 4px;
    margin-top: 10px;
    display: inline-block;
    font-size: 13px;
    transition: background-color 0.3s;
    border: none;
    text-decoration: none;
    font-weight: 500;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.debug-toggle:hover {
    background-color: #303f9f;
    box-shadow: 0 3px 5px rgba(0,0,0,0.2);
}
            """

            # Create CSS file
            with open(css_file_path, "w") as css_file:
                css_file.write(css_content)

            # Generate HTML report content with enhanced head section
            html_content = f"""<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Test Execution Report</title>
                <link rel="preconnect" href="https://fonts.googleapis.com">
                <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=Roboto:wght@400;500&family=Roboto+Mono&display=swap" rel="stylesheet">
                <link rel="stylesheet" href="assets/style.css">
                <script>
                    function toggleDebug(id) {{
                        var element = document.getElementById(id);
                        var link = document.getElementById('link-' + id);

                        if (element.style.maxHeight === "0px" || element.style.maxHeight === "") {{
                            element.style.maxHeight = "500px"; // Allow more space for debug info
                            element.style.opacity = "1";
                            link.textContent = "Hide Debug Info";
                        }} else {{
                            element.style.maxHeight = "0px";
                            element.style.opacity = "0";
                            link.textContent = "Show Debug Info";
                        }}
                    }}
                </script>
            </head>
            <body>
                <h1>Test Execution Report</h1>
                <div class="report-timestamp">
                    Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>

                <div class="summary-container">
                    <h2>Test Summary</h2>
                    <div class="summary-box total-box">
                        <div class="summary-value">{total}</div>
                        <div class="summary-label">TOTAL</div>
                    </div>
                    <div class="summary-box pass-box">
                        <div class="summary-value">{passed}</div>
                        <div class="summary-label">PASSED</div>
                    </div>
                    <div class="summary-box fail-box">
                        <div class="summary-value">{failed + error}</div>
                        <div class="summary-label">FAILED</div>
                    </div>
                    <div class="summary-box skip-box">
                        <div class="summary-value">{skipped}</div>
                        <div class="summary-label">SKIPPED</div>
                    </div>
                </div>

                <div class="results-container">
                    <h2>Detailed Test Results</h2>
                    <table>
                        <tr>
                            <th>Test Name</th>
                            <th>Status</th>
                            <th>Duration (s)</th>
                            <th>Error Message</th>
                        </tr>
            """

            # Process all tests and add rows to the results table
            categories = [
                ("passed", "PASS", "passed"),
                ("failed", "FAIL", "failed"),
                ("skipped", "SKIP", "skipped"),
                ("error", "ERROR", "error")
            ]

            test_counter = 0
            for category, status_text, css_class in categories:
                for rep in reporter.stats.get(category, []):
                    test_counter += 1
                    test_name = rep.nodeid

                    # Get duration if available
                    duration = ""
                    if hasattr(rep, "duration"):
                        duration = f"{rep.duration:.3f}"

                    # Get error message if available
                    error_msg = ""
                    if hasattr(rep, "longrepr") and rep.longrepr:
                        error_text = str(rep.longrepr)
                        for line in error_text.split('\n'):
                            if 'Error:' in line or 'AssertionError:' in line:
                                error_msg = line.strip()
                                break

                    # Get full debug information
                    debug_info = ""
                    if hasattr(rep, "longrepr") and rep.longrepr:
                        debug_info = str(rep.longrepr)
                    elif category == "skipped" and hasattr(rep, "longrepr"):
                        # For skipped tests, show the skip reason
                        debug_info = str(rep.longrepr[2]) if isinstance(rep.longrepr, tuple) and len(
                            rep.longrepr) > 2 else ""

                    html_content += f"""
                        <tr class="{css_class}">
                            <td>{test_name}</td>
                            <td>{status_text}</td>
                            <td>{duration}</td>
                            <td class="error-message">
                                {error_msg}
                                <button id="link-debug-{test_counter}" class="debug-toggle" onclick="toggleDebug('debug-{test_counter}')">Show Debug Info</button>
                                <div id="debug-{test_counter}" class="debug-info">
                                    {debug_info.replace('<', '&lt;').replace('>', '&gt;')}
                                </div>
                            </td>
                        </tr>
                    """

            # Close the HTML structure
            html_content += """
                    </table>
                </div>
            </body>
            </html>
            """

            # Write HTML report
            with open(html_report_path, "w") as html_file:
                html_file.write(html_content)

            # Create ZIP file with the report and assets in the Reports folder
            zip_filename = f"test_report_{timestamp}.zip"
            zip_filepath = os.path.join(reports_dir, zip_filename)
            with zipfile.ZipFile(zip_filepath, 'w') as zipf:
                # Add report.html - using relative paths within the zip
                zipf.write(html_report_path, arcname="report.html")
                # Add style.css - using relative paths within the zip
                zipf.write(css_file_path, arcname="assets/style.css")

            print(f"\n=== HTML Test Report created: {html_report_path} ===")
            print(f"=== ZIP archive created: {zip_filepath} ===")

            print("\n=== Running JIRA issue matching utility ===")

            # Path to your JIRA utility script  #JIRA_PATH
            jira_script_path = os.path.join(os.path.dirname(__file__),
                                            "/Users/jagdishpatil/Desktop/python_projects/spaCy/utilities/jiraUtils.py")

            # Make sure the path is absolute and normalized
            jira_script_path = os.path.abspath(os.path.normpath(jira_script_path))

            try:
                # Run the JIRA utility script
                subprocess.run([sys.executable, jira_script_path], check=True)
                print("JIRA issue matching completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error running JIRA utility: {e}")
            except Exception as e:
                print(f"Unexpected error running JIRA utility: {e}")



# THIS IS COMPLETED WITH GOOD DEBUGGIBG
# def pytest_sessionfinish(session, exitstatus):
#     """
#     Called after the test run finishes. Writes only test case names and
#     assertion error messages to a text file, then runs the JIRA utility.
#     """
#     if exitstatus != 0:  # Non-zero exit status means some tests failed
#         with open("failed_tests.txt", "w") as f:
#             f.write("=== FAILED TESTS SUMMARY ===\n\n")
#
#             failed_count = 0
#             reporter = session.config.pluginmanager.get_plugin('terminalreporter')
#             if reporter:
#                 for test_id, reports in reporter.stats.items():
#                     if test_id == 'failed':
#                         for rep in reports:
#                             failed_count += 1
#                             # Get just the test name
#                             test_name = rep.nodeid
#                             f.write(f"Test: {test_name}\n")
#
#                             # Extract only the AssertionError message
#                             if hasattr(rep, "longrepr"):
#                                 error_text = str(rep.longrepr)
#                                 # Find the AssertionError line
#                                 for line in error_text.split('\n'):
#                                     if 'AssertionError:' in line:
#                                         # Extract just the message part
#                                         error_message = line.strip()
#                                         f.write(f"Error: {error_message}\n")
#                                         break
#
#                             f.write("-" * 80 + "\n\n")
#
#             f.write(f"Total Failed Tests: {failed_count}\n")
#
#         # If there are failed tests, run the JIRA utility
#         if failed_count > 0:
#             # Create timestamp for unique report naming
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#             # Create the report directory structure
#             reports_dir = "Reports"
#             assets_dir = os.path.join(reports_dir, "assets")
#
#             # Create directories if they don't exist
#             os.makedirs(reports_dir, exist_ok=True)
#             os.makedirs(assets_dir, exist_ok=True)
#
#             # Path for the HTML report and CSS file
#             html_report_path = os.path.join(reports_dir, "report.html")
#             css_file_path = os.path.join(assets_dir, "style.css")
#
#             # Calculate test statistics
#             total = len(reporter.stats.get("passed", [])) + len(reporter.stats.get("failed", [])) + len(
#                 reporter.stats.get("skipped", [])) + len(reporter.stats.get("error", []))
#             passed = len(reporter.stats.get("passed", []))
#             failed = len(reporter.stats.get("failed", []))
#             error = len(reporter.stats.get("error", []))
#             skipped = len(reporter.stats.get("skipped", []))
#
#             # Create CSS content
#             css_content = """
#             body {
#                 font-family: Arial, sans-serif;
#                 margin: 20px;
#                 background-color: #f5f5f5;
#             }
#
#             h1, h2, h3 {
#                 color: #333;
#             }
#
#             h1 {
#                 text-align: center;
#                 padding: 10px;
#                 background-color: #4CAF50;
#                 color: white;
#                 border-radius: 5px;
#             }
#
#             .report-timestamp {
#                 text-align: center;
#                 color: #666;
#                 margin-bottom: 20px;
#             }
#
#             .summary-container {
#                 padding: 15px;
#                 background-color: white;
#                 border-radius: 5px;
#                 box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#                 margin-bottom: 20px;
#             }
#
#             .results-container {
#                 padding: 15px;
#                 background-color: white;
#                 border-radius: 5px;
#                 box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#             }
#
#             table {
#                 width: 100%;
#                 border-collapse: collapse;
#                 margin-bottom: 20px;
#             }
#
#             th, td {
#                 padding: 10px;
#                 text-align: left;
#                 border-bottom: 1px solid #ddd;
#             }
#
#             th {
#                 background-color: #f2f2f2;
#                 font-weight: bold;
#             }
#
#             tr:hover {
#                 background-color: #f5f5f5;
#             }
#
#             .passed {
#                 background-color: #dff0d8;
#                 color: #3c763d;
#             }
#
#             .failed {
#                 background-color: #f2dede;
#                 color: #a94442;
#             }
#
#             .skipped {
#                 background-color: #fcf8e3;
#                 color: #8a6d3b;
#             }
#
#             .error {
#                 background-color: #f2dede;
#                 color: #a94442;
#             }
#
#             .summary-box {
#                 display: inline-block;
#                 width: 22%;
#                 margin: 0 1%;
#                 padding: 10px;
#                 text-align: center;
#                 border-radius: 5px;
#                 box-shadow: 0 1px 3px rgba(0,0,0,0.1);
#             }
#
#             .total-box {
#                 background-color: #f5f5f5;
#                 border: 1px solid #ddd;
#             }
#
#             .pass-box {
#                 background-color: #dff0d8;
#                 border: 1px solid #d6e9c6;
#             }
#
#             .fail-box {
#                 background-color: #f2dede;
#                 border: 1px solid #ebccd1;
#             }
#
#             .skip-box {
#                 background-color: #fcf8e3;
#                 border: 1px solid #faebcc;
#             }
#
#             .summary-value {
#                 font-size: 24px;
#                 font-weight: bold;
#                 margin: 5px 0;
#             }
#
#             .summary-label {
#                 font-size: 14px;
#                 color: #666;
#             }
#
#             .error-message {
#                 font-family: monospace;
#                 white-space: pre-wrap;
#                 color: #a94442;
#             }
#
#             .debug-info {
#                 font-family: 'Consolas', 'Monaco', monospace;
#                 font-size: 13px;
#                 line-height: 1.5;
#                 background-color: #2d2d2d;
#                 color: #e0e0e0;
#                 border-radius: 5px;
#                 padding: 15px;
#                 margin-top: 10px;
#                 white-space: pre-wrap;
#                 overflow-x: auto;
#                 max-height: 0px;
#                 opacity: 0;
#                 transition: max-height 0.5s ease-in-out, opacity 0.4s ease-in-out;
#                 word-wrap: break-word;
#                 box-shadow: 0 3px 6px rgba(0,0,0,0.16);
#             }
#
#             .debug-toggle {
#                 cursor: pointer;
#                 background-color: #4b8bf4;
#                 color: white;
#                 padding: 6px 12px;
#                 border-radius: 4px;
#                 margin-top: 8px;
#                 display: inline-block;
#                 font-size: 13px;
#                 transition: background-color 0.3s;
#                 border: none;
#                 text-decoration: none;
#             }
#
#             .debug-toggle:hover {
#                 background-color: #3a78de;
#             }
#             """
#
#             # Create CSS file
#             with open(css_file_path, "w") as css_file:
#                 css_file.write(css_content)
#
#             # Generate HTML report content
#             html_content = f"""<!DOCTYPE html>
#             <html lang="en">
#             <head>
#                 <meta charset="UTF-8">
#                 <meta name="viewport" content="width=device-width, initial-scale=1.0">
#                 <title>Test Execution Report</title>
#                 <link rel="stylesheet" href="assets/style.css">
#                 <script>
#                     function toggleDebug(id) {{
#                         var element = document.getElementById(id);
#                         var link = document.getElementById('link-' + id);
#
#                         if (element.style.maxHeight === "0px" || element.style.maxHeight === "") {{
#                             element.style.maxHeight = "500px"; // Allow more space for debug info
#                             element.style.opacity = "1";
#                             link.textContent = "Hide Debug Info";
#                         }} else {{
#                             element.style.maxHeight = "0px";
#                             element.style.opacity = "0";
#                             link.textContent = "Show Debug Info";
#                         }}
#                     }}
#                 </script>
#             </head>
#             <body>
#                 <h1>Test Execution Report</h1>
#                 <div class="report-timestamp">
#                     Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#                 </div>
#
#                 <div class="summary-container">
#                     <h2>Test Summary</h2>
#                     <div class="summary-box total-box">
#                         <div class="summary-value">{total}</div>
#                         <div class="summary-label">TOTAL</div>
#                     </div>
#                     <div class="summary-box pass-box">
#                         <div class="summary-value">{passed}</div>
#                         <div class="summary-label">PASSED</div>
#                     </div>
#                     <div class="summary-box fail-box">
#                         <div class="summary-value">{failed + error}</div>
#                         <div class="summary-label">FAILED</div>
#                     </div>
#                     <div class="summary-box skip-box">
#                         <div class="summary-value">{skipped}</div>
#                         <div class="summary-label">SKIPPED</div>
#                     </div>
#                 </div>
#
#                 <div class="results-container">
#                     <h2>Detailed Test Results</h2>
#                     <table>
#                         <tr>
#                             <th>Test Name</th>
#                             <th>Status</th>
#                             <th>Duration (s)</th>
#                             <th>Error Message</th>
#                         </tr>
#             """
#
#             # Process all tests and add rows to the results table
#             categories = [
#                 ("passed", "PASS", "passed"),
#                 ("failed", "FAIL", "failed"),
#                 ("skipped", "SKIP", "skipped"),
#                 ("error", "ERROR", "error")
#             ]
#
#             test_counter = 0
#             for category, status_text, css_class in categories:
#                 for rep in reporter.stats.get(category, []):
#                     test_counter += 1
#                     test_name = rep.nodeid
#
#                     # Get duration if available
#                     duration = ""
#                     if hasattr(rep, "duration"):
#                         duration = f"{rep.duration:.3f}"
#
#                     # Get error message if available
#                     error_msg = ""
#                     if hasattr(rep, "longrepr") and rep.longrepr:
#                         error_text = str(rep.longrepr)
#                         for line in error_text.split('\n'):
#                             if 'Error:' in line or 'AssertionError:' in line:
#                                 error_msg = line.strip()
#                                 break
#
#                     # Get full debug information
#                     debug_info = ""
#                     if hasattr(rep, "longrepr") and rep.longrepr:
#                         debug_info = str(rep.longrepr)
#                     elif category == "skipped" and hasattr(rep, "longrepr"):
#                         # For skipped tests, show the skip reason
#                         debug_info = str(rep.longrepr[2]) if isinstance(rep.longrepr, tuple) and len(
#                             rep.longrepr) > 2 else ""
#
#                     html_content += f"""
#                         <tr class="{css_class}">
#                             <td>{test_name}</td>
#                             <td>{status_text}</td>
#                             <td>{duration}</td>
#                             <td class="error-message">
#                                 {error_msg}
#                                 <button id="link-debug-{test_counter}" class="debug-toggle" onclick="toggleDebug('debug-{test_counter}')">Show Debug Info</button>
#                                 <div id="debug-{test_counter}" class="debug-info">
#                                     {debug_info.replace('<', '&lt;').replace('>', '&gt;')}
#                                 </div>
#                             </td>
#                         </tr>
#                     """
#
#             # Close the HTML structure
#             html_content += """
#                     </table>
#                 </div>
#             </body>
#             </html>
#             """
#
#             # Write HTML report
#             with open(html_report_path, "w") as html_file:
#                 html_file.write(html_content)
#
#             # Create ZIP file with the report and assets in the Reports folder
#             zip_filename = f"test_report_{timestamp}.zip"
#             zip_filepath = os.path.join(reports_dir, zip_filename)
#             with zipfile.ZipFile(zip_filepath, 'w') as zipf:
#                 # Add report.html - using relative paths within the zip
#                 zipf.write(html_report_path, arcname="report.html")
#                 # Add style.css - using relative paths within the zip
#                 zipf.write(css_file_path, arcname="assets/style.css")
#
#             print(f"\n=== HTML Test Report created: {html_report_path} ===")
#             print(f"=== ZIP archive created: {zip_filepath} ===")
#
#             # # Create ZIP file with the report and assets
#             # zip_filename = f"test_report_{timestamp}.zip"
#             # with zipfile.ZipFile(zip_filename, 'w') as zipf:
#             #     # Add report.html
#             #     zipf.write(html_report_path, arcname="Reports/report.html")
#             #     # Add style.css
#             #     zipf.write(css_file_path, arcname="Reports/assets/style.css")
#
#             # print(f"\n=== HTML Test Report created: {html_report_path} ===")
#             # print(f"=== ZIP archive created: {zip_filename} ===")
#
#             print("\n=== Running JIRA issue matching utility ===")
#
#             # Path to your JIRA utility script
#             jira_script_path = os.path.join(os.path.dirname(__file__),
#                                             "/home/deepak/Desktop/Selenium_JIRA/Selenium_JIRA/Python_Selenium_Framework/utilities/jiraUtils.py")
#
#             # Make sure the path is absolute and normalized
#             jira_script_path = os.path.abspath(os.path.normpath(jira_script_path))
#
#             try:
#                 # Run the JIRA utility script
#                 subprocess.run([sys.executable, jira_script_path], check=True)
#                 print("JIRA issue matching completed successfully.")
#             except subprocess.CalledProcessError as e:
#                 print(f"Error running JIRA utility: {e}")
#             except Exception as e:
#                 print(f"Unexpected error running JIRA utility: {e}")


# SOMEWHAT WORKING
# def pytest_sessionfinish(session, exitstatus):
#     """
#     Called after the test run finishes. Writes only test case names and
#     assertion error messages to a text file, then runs the JIRA utility.
#     """
#     if exitstatus != 0:  # Non-zero exit status means some tests failed
#         with open("failed_tests.txt", "w") as f:
#             f.write("=== FAILED TESTS SUMMARY ===\n\n")
#
#             failed_count = 0
#             reporter = session.config.pluginmanager.get_plugin('terminalreporter')
#             if reporter:
#                 for test_id, reports in reporter.stats.items():
#                     if test_id == 'failed':
#                         for rep in reports:
#                             failed_count += 1
#                             # Get just the test name
#                             test_name = rep.nodeid
#                             f.write(f"Test: {test_name}\n")
#
#                             # Extract only the AssertionError message
#                             if hasattr(rep, "longrepr"):
#                                 error_text = str(rep.longrepr)
#                                 # Find the AssertionError line
#                                 for line in error_text.split('\n'):
#                                     if 'AssertionError:' in line:
#                                         # Extract just the message part
#                                         error_message = line.strip()
#                                         f.write(f"Error: {error_message}\n")
#                                         break
#
#                             f.write("-" * 80 + "\n\n")
#
#             f.write(f"Total Failed Tests: {failed_count}\n")
#
#         # If there are failed tests, run the JIRA utility
#         if failed_count > 0:
#             # Create timestamp for unique report naming
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#             # Create the report directory structure
#             reports_dir = "Reports"
#             assets_dir = os.path.join(reports_dir, "assets")
#
#             # Create directories if they don't exist
#             os.makedirs(reports_dir, exist_ok=True)
#             os.makedirs(assets_dir, exist_ok=True)
#
#             # Path for the HTML report and CSS file
#             html_report_path = os.path.join(reports_dir, "report.html")
#             css_file_path = os.path.join(assets_dir, "style.css")
#
#             # Calculate test statistics
#             total = len(reporter.stats.get("passed", [])) + len(reporter.stats.get("failed", [])) + len(
#                 reporter.stats.get("skipped", [])) + len(reporter.stats.get("error", []))
#             passed = len(reporter.stats.get("passed", []))
#             failed = len(reporter.stats.get("failed", []))
#             error = len(reporter.stats.get("error", []))
#             skipped = len(reporter.stats.get("skipped", []))
#
#             # Create CSS content
#             css_content = """
#             body {
#                 font-family: Arial, sans-serif;
#                 margin: 20px;
#                 background-color: #f5f5f5;
#             }
#
#             h1, h2, h3 {
#                 color: #333;
#             }
#
#             h1 {
#                 text-align: center;
#                 padding: 10px;
#                 background-color: #4CAF50;
#                 color: white;
#                 border-radius: 5px;
#             }
#
#             .report-timestamp {
#                 text-align: center;
#                 color: #666;
#                 margin-bottom: 20px;
#             }
#
#             .summary-container {
#                 padding: 15px;
#                 background-color: white;
#                 border-radius: 5px;
#                 box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#                 margin-bottom: 20px;
#             }
#
#             .results-container {
#                 padding: 15px;
#                 background-color: white;
#                 border-radius: 5px;
#                 box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#             }
#
#             table {
#                 width: 100%;
#                 border-collapse: collapse;
#                 margin-bottom: 20px;
#             }
#
#             th, td {
#                 padding: 10px;
#                 text-align: left;
#                 border-bottom: 1px solid #ddd;
#             }
#
#             th {
#                 background-color: #f2f2f2;
#                 font-weight: bold;
#             }
#
#             tr:hover {
#                 background-color: #f5f5f5;
#             }
#
#             .passed {
#                 background-color: #dff0d8;
#                 color: #3c763d;
#             }
#
#             .failed {
#                 background-color: #f2dede;
#                 color: #a94442;
#             }
#
#             .skipped {
#                 background-color: #fcf8e3;
#                 color: #8a6d3b;
#             }
#
#             .error {
#                 background-color: #f2dede;
#                 color: #a94442;
#             }
#
#             .summary-box {
#                 display: inline-block;
#                 width: 22%;
#                 margin: 0 1%;
#                 padding: 10px;
#                 text-align: center;
#                 border-radius: 5px;
#                 box-shadow: 0 1px 3px rgba(0,0,0,0.1);
#             }
#
#             .total-box {
#                 background-color: #f5f5f5;
#                 border: 1px solid #ddd;
#             }
#
#             .pass-box {
#                 background-color: #dff0d8;
#                 border: 1px solid #d6e9c6;
#             }
#
#             .fail-box {
#                 background-color: #f2dede;
#                 border: 1px solid #ebccd1;
#             }
#
#             .skip-box {
#                 background-color: #fcf8e3;
#                 border: 1px solid #faebcc;
#             }
#
#             .summary-value {
#                 font-size: 24px;
#                 font-weight: bold;
#                 margin: 5px 0;
#             }
#
#             .summary-label {
#                 font-size: 14px;
#                 color: #666;
#             }
#
#             .error-message {
#                 font-family: monospace;
#                 white-space: pre-wrap;
#                 color: #a94442;
#             }
#
#             .debug-info {
#                 font-family: monospace;
#                 font-size: 12px;
#                 background-color: #f8f8f8;
#                 border: 1px solid #ddd;
#                 padding: 10px;
#                 margin-top: 5px;
#                 white-space: pre-wrap;
#                 color: #666;
#                 max-height: 200px;
#                 overflow-y: auto;
#             }
#
#             .debug-toggle {
#                 cursor: pointer;
#                 color: #337ab7;
#                 text-decoration: underline;
#                 margin-top: 5px;
#                 display: block;
#             }
#             """
#
#             # Create CSS file
#             with open(css_file_path, "w") as css_file:
#                 css_file.write(css_content)
#
#             # Generate HTML report content
#             html_content = f"""<!DOCTYPE html>
#             <html lang="en">
#             <head>
#                 <meta charset="UTF-8">
#                 <meta name="viewport" content="width=device-width, initial-scale=1.0">
#                 <title>Test Execution Report</title>
#                 <link rel="stylesheet" href="assets/style.css">
#                 <script>
#                     function toggleDebug(id) {{
#                         var element = document.getElementById(id);
#                         if (element.style.display === "none") {{
#                             element.style.display = "block";
#                         }} else {{
#                             element.style.display = "none";
#                         }}
#                     }}
#                 </script>
#             </head>
#             <body>
#                 <h1>Test Execution Report</h1>
#                 <div class="report-timestamp">
#                     Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#                 </div>
#
#                 <div class="summary-container">
#                     <h2>Test Summary</h2>
#                     <div class="summary-box total-box">
#                         <div class="summary-value">{total}</div>
#                         <div class="summary-label">TOTAL</div>
#                     </div>
#                     <div class="summary-box pass-box">
#                         <div class="summary-value">{passed}</div>
#                         <div class="summary-label">PASSED</div>
#                     </div>
#                     <div class="summary-box fail-box">
#                         <div class="summary-value">{failed + error}</div>
#                         <div class="summary-label">FAILED</div>
#                     </div>
#                     <div class="summary-box skip-box">
#                         <div class="summary-value">{skipped}</div>
#                         <div class="summary-label">SKIPPED</div>
#                     </div>
#                 </div>
#
#                 <div class="results-container">
#                     <h2>Detailed Test Results</h2>
#                     <table>
#                         <tr>
#                             <th>Test Name</th>
#                             <th>Status</th>
#                             <th>Duration (s)</th>
#                             <th>Error Message</th>
#                         </tr>
#             """
#
#             # Process all tests and add rows to the results table
#             categories = [
#                 ("passed", "PASS", "passed"),
#                 ("failed", "FAIL", "failed"),
#                 ("skipped", "SKIP", "skipped"),
#                 ("error", "ERROR", "error")
#             ]
#
#             test_counter = 0
#             for category, status_text, css_class in categories:
#                 for rep in reporter.stats.get(category, []):
#                     test_counter += 1
#                     test_name = rep.nodeid
#
#                     # Get duration if available
#                     duration = ""
#                     if hasattr(rep, "duration"):
#                         duration = f"{rep.duration:.3f}"
#
#                     # Get error message if available
#                     error_msg = ""
#                     if hasattr(rep, "longrepr") and rep.longrepr:
#                         error_text = str(rep.longrepr)
#                         for line in error_text.split('\n'):
#                             if 'Error:' in line or 'AssertionError:' in line:
#                                 error_msg = line.strip()
#                                 break
#
#                     # Get full debug information
#                     debug_info = ""
#                     if hasattr(rep, "longrepr") and rep.longrepr:
#                         debug_info = str(rep.longrepr)
#                     elif category == "skipped" and hasattr(rep, "longrepr"):
#                         # For skipped tests, show the skip reason
#                         debug_info = str(rep.longrepr[2]) if isinstance(rep.longrepr, tuple) and len(
#                             rep.longrepr) > 2 else ""
#
#                     html_content += f"""
#                         <tr class="{css_class}">
#                             <td>{test_name}</td>
#                             <td>{status_text}</td>
#                             <td>{duration}</td>
#                             <td class="error-message">
#                                 {error_msg}
#                                 <a class="debug-toggle" onclick="toggleDebug('debug-{test_counter}')">Show/Hide Debug Info</a>
#                                 <div id="debug-{test_counter}" class="debug-info" style="display: none;">
#                                     {debug_info.replace('<', '&lt;').replace('>', '&gt;')}
#                                 </div>
#                             </td>
#                         </tr>
#                     """
#
#             # Close the HTML structure
#             html_content += """
#                     </table>
#                 </div>
#             </body>
#             </html>
#             """
#
#             # Write HTML report
#             with open(html_report_path, "w") as html_file:
#                 html_file.write(html_content)
#
#             # Create ZIP file with the report and assets
#             zip_filename = f"test_report_{timestamp}.zip"
#             with zipfile.ZipFile(zip_filename, 'w') as zipf:
#                 # Add report.html
#                 zipf.write(html_report_path, arcname="Reports/report.html")
#                 # Add style.css
#                 zipf.write(css_file_path, arcname="Reports/assets/style.css")
#
#             print(f"\n=== HTML Test Report created: {html_report_path} ===")
#             print(f"=== ZIP archive created: {zip_filename} ===")
#
#             print("\n=== Running JIRA issue matching utility ===")
#
#             # Path to your JIRA utility script
#             jira_script_path = os.path.join(os.path.dirname(__file__),
#                                             "/home/deepak/Desktop/Selenium_JIRA/Selenium_JIRA/Python_Selenium_Framework/utilities/jiraUtils.py")
#
#             # Make sure the path is absolute and normalized
#             jira_script_path = os.path.abspath(os.path.normpath(jira_script_path))
#
#             try:
#                 # Run the JIRA utility script
#                 subprocess.run([sys.executable, jira_script_path], check=True)
#                 print("JIRA issue matching completed successfully.")
#             except subprocess.CalledProcessError as e:
#                 print(f"Error running JIRA utility: {e}")
#             except Exception as e:
#                 print(f"Unexpected error running JIRA utility: {e}")









# WORKING 100%

# def pytest_sessionfinish(session, exitstatus):
#     """
#     Called after the test run finishes. Writes only test case names and
#     assertion error messages to a text file, then runs the JIRA utility.
#     """
#     if exitstatus != 0:  # Non-zero exit status means some tests failed
#         with open("failed_tests.txt", "w") as f:
#             f.write("=== FAILED TESTS SUMMARY ===\n\n")
#
#             failed_count = 0
#             reporter = session.config.pluginmanager.get_plugin('terminalreporter')
#             if reporter:
#                 for test_id, reports in reporter.stats.items():
#                     if test_id == 'failed':
#                         for rep in reports:
#                             failed_count += 1
#                             # Get just the test name
#                             test_name = rep.nodeid
#                             f.write(f"Test: {test_name}\n")
#
#                             # Extract only the AssertionError message
#                             if hasattr(rep, "longrepr"):
#                                 error_text = str(rep.longrepr)
#                                 # Find the AssertionError line
#                                 for line in error_text.split('\n'):
#                                     if 'AssertionError:' in line:
#                                         # Extract just the message part
#                                         error_message = line.strip()
#                                         f.write(f"Error: {error_message}\n")
#                                         break
#
#                             f.write("-" * 80 + "\n\n")
#
#             f.write(f"Total Failed Tests: {failed_count}\n")
#
#         # If there are failed tests, run the JIRA utility
#         if failed_count > 0:
#             # Create timestamp for unique report naming
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#             # Create the report directory structure
#             reports_dir = "Reports"
#             assets_dir = os.path.join(reports_dir, "assets")
#
#             # Create directories if they don't exist
#             os.makedirs(reports_dir, exist_ok=True)
#             os.makedirs(assets_dir, exist_ok=True)
#
#             # Path for the HTML report and CSS file
#             html_report_path = os.path.join(reports_dir, "report.html")
#             css_file_path = os.path.join(assets_dir, "style.css")
#
#             # Calculate test statistics
#             total = len(reporter.stats.get("passed", [])) + len(reporter.stats.get("failed", [])) + len(
#                 reporter.stats.get("skipped", [])) + len(reporter.stats.get("error", []))
#             passed = len(reporter.stats.get("passed", []))
#             failed = len(reporter.stats.get("failed", []))
#             error = len(reporter.stats.get("error", []))
#             skipped = len(reporter.stats.get("skipped", []))
#
#             # Create CSS content
#             css_content = """
#             body {
#                 font-family: Arial, sans-serif;
#                 margin: 20px;
#                 background-color: #f5f5f5;
#             }
#
#             h1, h2, h3 {
#                 color: #333;
#             }
#
#             h1 {
#                 text-align: center;
#                 padding: 10px;
#                 background-color: #4CAF50;
#                 color: white;
#                 border-radius: 5px;
#             }
#
#             .report-timestamp {
#                 text-align: center;
#                 color: #666;
#                 margin-bottom: 20px;
#             }
#
#             .summary-container {
#                 padding: 15px;
#                 background-color: white;
#                 border-radius: 5px;
#                 box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#                 margin-bottom: 20px;
#             }
#
#             .results-container {
#                 padding: 15px;
#                 background-color: white;
#                 border-radius: 5px;
#                 box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#             }
#
#             table {
#                 width: 100%;
#                 border-collapse: collapse;
#                 margin-bottom: 20px;
#             }
#
#             th, td {
#                 padding: 10px;
#                 text-align: left;
#                 border-bottom: 1px solid #ddd;
#             }
#
#             th {
#                 background-color: #f2f2f2;
#                 font-weight: bold;
#             }
#
#             tr:hover {
#                 background-color: #f5f5f5;
#             }
#
#             .passed {
#                 background-color: #dff0d8;
#                 color: #3c763d;
#             }
#
#             .failed {
#                 background-color: #f2dede;
#                 color: #a94442;
#             }
#
#             .skipped {
#                 background-color: #fcf8e3;
#                 color: #8a6d3b;
#             }
#
#             .error {
#                 background-color: #f2dede;
#                 color: #a94442;
#             }
#
#             .summary-box {
#                 display: inline-block;
#                 width: 22%;
#                 margin: 0 1%;
#                 padding: 10px;
#                 text-align: center;
#                 border-radius: 5px;
#                 box-shadow: 0 1px 3px rgba(0,0,0,0.1);
#             }
#
#             .total-box {
#                 background-color: #f5f5f5;
#                 border: 1px solid #ddd;
#             }
#
#             .pass-box {
#                 background-color: #dff0d8;
#                 border: 1px solid #d6e9c6;
#             }
#
#             .fail-box {
#                 background-color: #f2dede;
#                 border: 1px solid #ebccd1;
#             }
#
#             .skip-box {
#                 background-color: #fcf8e3;
#                 border: 1px solid #faebcc;
#             }
#
#             .summary-value {
#                 font-size: 24px;
#                 font-weight: bold;
#                 margin: 5px 0;
#             }
#
#             .summary-label {
#                 font-size: 14px;
#                 color: #666;
#             }
#
#             .error-message {
#                 font-family: monospace;
#                 white-space: pre-wrap;
#                 color: #a94442;
#             }
#             """
#
#             # Create CSS file
#             with open(css_file_path, "w") as css_file:
#                 css_file.write(css_content)
#
#             # Generate HTML report content
#             html_content = f"""<!DOCTYPE html>
#             <html lang="en">
#             <head>
#                 <meta charset="UTF-8">
#                 <meta name="viewport" content="width=device-width, initial-scale=1.0">
#                 <title>Test Execution Report</title>
#                 <link rel="stylesheet" href="assets/style.css">
#             </head>
#             <body>
#                 <h1>Test Execution Report</h1>
#                 <div class="report-timestamp">
#                     Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#                 </div>
#
#                 <div class="summary-container">
#                     <h2>Test Summary</h2>
#                     <div class="summary-box total-box">
#                         <div class="summary-value">{total}</div>
#                         <div class="summary-label">TOTAL</div>
#                     </div>
#                     <div class="summary-box pass-box">
#                         <div class="summary-value">{passed}</div>
#                         <div class="summary-label">PASSED</div>
#                     </div>
#                     <div class="summary-box fail-box">
#                         <div class="summary-value">{failed + error}</div>
#                         <div class="summary-label">FAILED</div>
#                     </div>
#                     <div class="summary-box skip-box">
#                         <div class="summary-value">{skipped}</div>
#                         <div class="summary-label">SKIPPED</div>
#                     </div>
#                 </div>
#
#                 <div class="results-container">
#                     <h2>Detailed Test Results</h2>
#                     <table>
#                         <tr>
#                             <th>Test Name</th>
#                             <th>Status</th>
#                             <th>Duration (s)</th>
#                             <th>Error Message</th>
#                         </tr>
#             """
#
#             # Process all tests and add rows to the results table
#             categories = [
#                 ("passed", "PASS", "passed"),
#                 ("failed", "FAIL", "failed"),
#                 ("skipped", "SKIP", "skipped"),
#                 ("error", "ERROR", "error")
#             ]
#
#             for category, status_text, css_class in categories:
#                 for rep in reporter.stats.get(category, []):
#                     test_name = rep.nodeid
#
#                     # Get duration if available
#                     duration = ""
#                     if hasattr(rep, "duration"):
#                         duration = f"{rep.duration:.3f}"
#
#                     # Get error message if available
#                     error_msg = ""
#                     if hasattr(rep, "longrepr") and rep.longrepr:
#                         error_text = str(rep.longrepr)
#                         for line in error_text.split('\n'):
#                             if 'Error:' in line or 'AssertionError:' in line:
#                                 error_msg = line.strip()
#                                 break
#
#                     html_content += f"""
#                         <tr class="{css_class}">
#                             <td>{test_name}</td>
#                             <td>{status_text}</td>
#                             <td>{duration}</td>
#                             <td class="error-message">{error_msg}</td>
#                         </tr>
#                     """
#
#             # Close the HTML structure
#             html_content += """
#                     </table>
#                 </div>
#             </body>
#             </html>
#             """
#
#             # Write HTML report
#             with open(html_report_path, "w") as html_file:
#                 html_file.write(html_content)
#
#             # Create ZIP file with the report and assets
#             zip_filename = f"test_report_{timestamp}.zip"
#             with zipfile.ZipFile(zip_filename, 'w') as zipf:
#                 # Add report.html
#                 zipf.write(html_report_path, arcname="Reports/report.html")
#                 # Add style.css
#                 zipf.write(css_file_path, arcname="Reports/assets/style.css")
#
#             print(f"\n=== HTML Test Report created: {html_report_path} ===")
#             print(f"=== ZIP archive created: {zip_filename} ===")
#
#             print("\n=== Running JIRA issue matching utility ===")
#
#             # Path to your JIRA utility script
#             jira_script_path = os.path.join(os.path.dirname(__file__), "/home/deepak/Desktop/Selenium_JIRA/Selenium_JIRA/Python_Selenium_Framework/utilities/jiraUtils.py")
#
#             # Make sure the path is absolute and normalized
#             jira_script_path = os.path.abspath(os.path.normpath(jira_script_path))
#
#             try:
#                 # Run the JIRA utility script
#                 subprocess.run([sys.executable, jira_script_path], check=True)
#                 print("JIRA issue matching completed successfully.")
#             except subprocess.CalledProcessError as e:
#                 print(f"Error running JIRA utility: {e}")
#             except Exception as e:
#                 print(f"Unexpected error running JIRA utility: {e}")



