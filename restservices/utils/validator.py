
def validate(output,temperature,top_p,max_tokens):
    try:
        temperature = float(temperature)
        if not(temperature > 0.0 and temperature < 2.0):
            output.status = "failure"
            output.errorCode = "50"
            output.errorMsg = "Temperature should be between 0.0 and 2.0"

        top_p = float(top_p)
        if not(top_p > 0.0 and top_p < 1.0):
            output.status = "failure"
            output.errorCode = "60"
            output.errorMsg = "top_p should be between 0.0 and 1.0"

        max_tokens = int(max_tokens)
        if not (max_tokens > 64 and max_tokens < 4096):
            output.status = "failure"
            output.errorCode = "70"
            output.errorMsg = "Max_tokens should be between 64 and 4096"

    except Exception as error:
        output.status = "failure"
        output.errorCode = "200"
        output.errorMsg = "Error in parsing input:"+str(error)


    return output