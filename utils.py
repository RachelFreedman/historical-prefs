def parse_responses(response_text):
    """Parse the standard Community Alignment response text into a dict of individual responses A, B, C, D"""
    responses = {}
    current_response = None
    current_text = []
    
    for line in response_text.split('\n'):
        if line.startswith('# Response A:'):
            if current_response:  # Save previous response
                responses[current_response] = '\n'.join(current_text).strip()
            current_response = 'A'
            current_text = []
        elif line.startswith('# Response B:'):
            if current_response:
                responses[current_response] = '\n'.join(current_text).strip()
            current_response = 'B'
            current_text = []
        elif line.startswith('# Response C:'):
            if current_response:
                responses[current_response] = '\n'.join(current_text).strip()
            current_response = 'C'
            current_text = []
        elif line.startswith('# Response D:'):
            if current_response:
                responses[current_response] = '\n'.join(current_text).strip()
            current_response = 'D'
            current_text = []
        elif current_response:  # Add content to current response
            current_text.append(line)
    
    # Save last response
    if current_response:
        responses[current_response] = '\n'.join(current_text).strip()
    
    return responses