def modify_latex(latex_code):
    # Split the LaTeX code into lines
    lines = latex_code.split("\n")

    # List to store the resulting lines
    result_lines = []

    # Add footnotesize environment
    result_lines.append("\\begin{footnotesize}")

    # Flags to center 'breakpoints' and 'error'
    breakpoints_flag = False
    error_flag = False

    # Variable to separate lines by No.
    prev_no = None

    for line in lines:
        # Center 'breakpoints' and 'error'
        if "breakpoint" in line:
            breakpoints_flag = True
        if "error" in line:
            error_flag = True
        if breakpoints_flag and "&" in line:
            line = line.replace("r}{breakpoint", "c}{breakpoint")
            breakpoints_flag = False
        if error_flag and "&" in line:
            line = line.replace("r}{error", "c}{error")
            error_flag = False

        # Separate lines by No. with \hline
        if "C" in line or "D" in line:
            no = line.split("&")[0].strip()
            if prev_no and prev_no.split(" ")[0] != no.split(" ")[0]:
                result_lines.append("\\hline")
            prev_no = no

            # Replace repeated No. with empty string
            if prev_no and no == prev_no:
                line = line.replace(no, " " * len(no))

        result_lines.append(line)

    # Close footnotesize environment
    result_lines.append("\\end{footnotesize}")

    # Join the result into a string
    modified_latex = "\n".join(result_lines)

    return modified_latex


if __name__ == "__main__":
    with open("result.tex", "r") as file:
        latex_code = file.read()

    # Call the function to modify the LaTeX code
    modified_latex_code = modify_latex(latex_code)

    # Write the modified LaTeX code to a file
    with open("result.tex", "w") as file:
        file.write(modified_latex_code)
