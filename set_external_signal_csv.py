import datetime as dt
import os
import re
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_FILE = os.path.join(HERE, "policy_demo.py")
CSV_PATH = "C:/Users/localadmin/Desktop/Shadab/alpamayo_outputs/alpamayo_offline_when_TuCsyBF3nHU.csv"


def main():
    if not os.path.isfile(POLICY_FILE):
        print("Could not find:", POLICY_FILE, file=sys.stderr)
        sys.exit(1)

    with open(POLICY_FILE, "r", encoding="utf-8") as input_file:
        content = input_file.read()

    # Timestamped non-.py backup: never overwrites an earlier backup and stays
    # out of grep/linter/git the way a second .py copy would not.
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{POLICY_FILE}.{timestamp}.bak"
    with open(backup_file, "w", encoding="utf-8") as output_file:
        output_file.write(content)

    new_line = 'EXTERNAL_SIGNAL_CSV = "' + CSV_PATH + '"'

    pattern = r'^EXTERNAL_SIGNAL_CSV\s*=\s*["\'].*?["\']\s*$'

    if re.search(pattern, content, flags=re.MULTILINE):
        # Use a callable replacement so backslashes in CSV_PATH (normal Windows
        # paths) are inserted literally instead of parsed as escape sequences.
        updated = re.sub(pattern, lambda _match: new_line, content, flags=re.MULTILINE)
        print("Updated existing EXTERNAL_SIGNAL_CSV line.")
    else:
        marker = "OUTPUT_DIR"
        marker_index = content.find(marker)

        if marker_index >= 0:
            line_end = content.find("\n", marker_index)
            updated = content[:line_end + 1] + "\n" + new_line + "\n" + content[line_end + 1:]
            print("Added EXTERNAL_SIGNAL_CSV after OUTPUT_DIR section.")
        else:
            updated = new_line + "\n\n" + content
            print("Added EXTERNAL_SIGNAL_CSV at the top of the file.")

    with open(POLICY_FILE, "w", encoding="utf-8") as output_file:
        output_file.write(updated)

    print("Backup saved as:", backup_file)
    print("CSV path set to:", CSV_PATH)


if __name__ == "__main__":
    main()
