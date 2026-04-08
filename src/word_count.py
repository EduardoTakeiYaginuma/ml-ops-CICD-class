def word_count_handler(event, context):
    """
    A simple function that counts
    the number of words in a string
    """
    try:
        msg = event["body"]
    except KeyError:
        return {"error": "no body"}

    n_words = len(msg.split())

    return {
        "len": len(msg),
        "words": n_words,
    }