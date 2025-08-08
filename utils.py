def wrap_text(text, font, max_width):
    """
    Wraps text to fit within a specified maximum width.
    
    Args:
        text (str): The input text to wrap.
        font (pygame.font.Font): The font object to use for measuring text size.
        max_width (int): The maximum width in pixels for each line.
        
    Returns:
        list: A list of strings, where each string is a wrapped line.
    """
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        if font.size(current_line + word + " ")[0] < max_width:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line.strip():
        lines.append(current_line.strip())
    return lines