import subprocess
import re

def get_installed_package_versions(requirements_file="requirements.txt"):
    """
    Reads a requirements.txt file and returns a dictionary of installed package 
    names and their versions.

    Args:
        requirements_file (str): The path to the requirements.txt file.

    Returns:
        dict: A dictionary where keys are package names and values are their versions.
              Returns an empty dictionary if there's an error.
    """

    installed_versions = {}
    try:
        with open(requirements_file, "r") as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        for package in packages:
            # Extract package name (handles cases with ==, >=, etc.)
            match = re.match(r"([a-zA-Z0-9_-]+)", package)
            if match:
                package_name = match.group(1)
            else:
                print(f"Warning: Could not parse package name from line: {package}")
                continue

            try:
                result = subprocess.run(
                    ["pip", "show", package_name], capture_output=True, text=True, check=True
                )
                output = result.stdout
                version_match = re.search(r"Version: (\S+)", output)
                if version_match:
                    installed_versions[package_name] = version_match.group(1)
                else:
                    print(f"Warning: Could not find version for {package_name}")

            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not find package {package_name}: {e}")
            except FileNotFoundError:
                print("Error: pip command not found.  Is pip installed?")
                return {}  # Exit if pip isn't found
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return {}  # Return empty dict on general error
    except FileNotFoundError:
        print(f"Error: requirements.txt file not found at {requirements_file}")
        return {}  # Return empty dict if file not found
    except Exception as e:
        print(f"An unexpected error occurred while processing the file: {e}")
        return {}  # Return empty dict on general file processing error

    return installed_versions

if __name__ == "__main__":
    installed_versions = get_installed_package_versions()
    if installed_versions:
        print("Installed Package Versions:")
        for package, version in installed_versions.items():
            print(f"- {package}: {version}")
    else:
        print("Could not retrieve package versions.")