# Define the input text
input_text = """
p_11(L=1)=  1.16861698391976054096090160828421402e-01
p_11(L=2)=  -1.58791218976969629026855467698538146e-02
p_11(L=3)=  -3.07593110758684387090375243384701003e-02
p_11(L=4)=  -3.67684393132183026620719243289343935e-02
p_11(L=5)=  -4.04690492227564739193475988707781569e-02
p_11(L=6)=  -4.32546872202945850700520235181476573e-02
"""

using DoubleFloats


# Extract the numbers using a regular expression
numbers = [parse(Double64, match(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", line).match) for line in split(input_text, '\n') if !isempty(line)]
# Print the array of numbers
println(numbers)
