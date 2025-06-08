import random

def get_paraphrase_context() -> str:
    templates = [
        "What type of object is shown in this image?",
        "What is the main subject of this photograph?",
        "Can you identify the object in this picture?",
        "What do you see in this image?",
        "What would you call the object displayed in this photo?",
        "Which category best describes the content of this image?",
        "Looking at this image, what object can you identify?",
        f"Among the given options, which one correctly identifies this {random.choice(['object', 'item', 'subject'])}?",
        "What is captured in this photograph?",
    ]
    
    descriptive_words = [
        "primary", "main", "central", "key", "fundamental", 
        "principal", "essential", "core", "basic", "dominant"
    ]
    
    action_words = [
        "shown", "displayed", "presented", "depicted", "illustrated",
        "captured", "featured", "represented", "appearing", "visible"
    ]
    
    dynamic_templates = [
        f"What is the {random.choice(descriptive_words)} object {random.choice(action_words)} in this image?",
        f"Can you identify the {random.choice(descriptive_words)} subject {random.choice(action_words)} here?",
        f"Which option correctly describes the {random.choice(descriptive_words)} element {random.choice(action_words)} in this photo?",
    ]
    
    all_templates = templates + dynamic_templates
    
    return random.choice(all_templates)

def paraphrase() -> str:
    """
    generate context paraphrasing to the question : "What type of object is presented in this image?"
    """

    prompt_type = random.random()
    
    if prompt_type < 0.4:  
        return get_paraphrase_context()
    
    elif prompt_type < 0.7:  
        detail_templates = [
            f"What specific type of object is {random.choice(['shown', 'displayed', 'visible'])} in this photograph?",
            f"Looking at the details of this image, which option best describes the {random.choice(['main subject', 'primary object', 'key element'])}?",
            f"Based on the visual characteristics, what would you identify as the {random.choice(['main', 'primary', 'key'])} subject?",
        ]
        return random.choice(detail_templates)
    
    else:  
        context_templates = [
            "Among these options, which one accurately describes the content of this image?",
            f"What {random.choice(['object', 'item', 'subject'])} can be identified in this particular photograph?",
            f"In this specific image, what is the {random.choice(['main', 'primary', 'central'])} focus?",
        ]
        return random.choice(context_templates)
    
def generate_unrelated_distraction_prompts():
    """
    generte unrelated distraction prompts that has no relation to the image
    """
    science_facts = [
        "The speed of light in vacuum is approximately 299,792 kilometers per second.",
        "Water molecules are composed of two hydrogen atoms and one oxygen atom.",
        "The human body contains approximately 37.2 trillion cells.",
        "Quantum mechanics describes the behavior of matter and energy at the atomic scale.",
        "The Earth's atmosphere is composed primarily of nitrogen and oxygen.",
        "DNA carries genetic instructions for the development of all known living organisms.",
        "The periodic table contains 118 known chemical elements.",
        "The average distance between Earth and the Sun is about 93 million miles.",
        "Einstein's theory of relativity connects space and time as spacetime.",
        "Photosynthesis converts light energy into chemical energy in plants.",
    ]
    
    historical_facts = [
        "The Great Wall of China was built over multiple dynasties spanning centuries.",
        "The printing press was invented by Johannes Gutenberg around 1440.",
        "The Industrial Revolution began in Britain in the late 18th century.",
        "Ancient Egypt's pyramids were built as tombs for pharaohs.",
        "The Roman Empire reached its peak territorial extent under Emperor Trajan.",
        "The Renaissance period marked a revival of classical learning in Europe.",
        "The Wright brothers achieved the first powered flight in 1903.",
        "The Cold War lasted from approximately 1947 to 1991.",
        "The Silk Road connected trade routes between East Asia and the Mediterranean.",
        "The Black Death pandemic occurred in medieval Europe during the 14th century.",
    ]
    
    daily_life = [
        "Regular exercise can improve both physical and mental health.",
        "A balanced diet should include various nutrients from different food groups.",
        "The average adult needs between 7 to 9 hours of sleep per night.",
        "Recycling helps reduce waste and conserve natural resources.",
        "Public transportation can help reduce carbon emissions in cities.",
        "Regular dental check-ups are recommended every six months.",
        "Staying hydrated is essential for maintaining good health.",
        "Social connections are important for mental well-being.",
        "Reading before bedtime can help improve sleep quality.",
        "Regular handwashing helps prevent the spread of diseases.",
    ]
    
    random_facts = [
        "Honey never spoils if stored properly.",
        "Penguins have specialized glands to remove salt from seawater.",
        "The shortest war in history lasted only 38 minutes.",
        "A day on Venus is longer than its year.",
        "Bananas are berries, but strawberries aren't.",
        "The first oranges weren't orange in color.",
        "Octopuses have three hearts and blue blood.",
        "Cows have best friends and get stressed when separated.",
        "A bolt of lightning is six times hotter than the sun's surface.",
        "The human nose can detect over one trillion different scents.",
    ]
    
    philosophical_questions = [
        "The nature of consciousness remains one of philosophy's greatest mysteries.",
        "Free will versus determinism is a fundamental philosophical debate.",
        "The concept of time is both universal and deeply personal.",
        "Reality might be different from what our senses perceive.",
        "The meaning of existence varies across different philosophical traditions.",
        "Knowledge and belief have complex relationships in epistemology.",
        "Ethics explores the nature of right and wrong in human behavior.",
        "The mind-body problem challenges our understanding of consciousness.",
        "Personal identity persists despite constant physical changes.",
        "The nature of truth remains a central philosophical question.",
    ]
    
    mathematical_concepts = [
        "The square root of negative one is the basis for complex numbers.",
        "Pi is an irrational number that continues infinitely without repetition.",
        "The Fibonacci sequence appears frequently in nature.",
        "Prime numbers are the building blocks of all natural numbers.",
        "The concept of zero revolutionized mathematical thinking.",
        "Euler's identity connects five fundamental mathematical constants.",
        "Parallel lines never intersect in Euclidean geometry.",
        "The golden ratio appears in art, architecture, and nature.",
        "Infinity is not a number but a mathematical concept.",
        "Probability theory helps understand random events mathematically.",
    ]

    incorrect_facts = [
        "The number zero was invented by ancient aliens to help build pyramids.",
        "Pi is exactly equal to 22/7 and never changes.",
        "Prime numbers always end in 7.",
        "Negative numbers were outlawed in Europe during the Renaissance.",
        "The square root of 2 is a rational number.",
        "Triangles can only have right angles if they are scalene.",
        "Infinity is the largest number known to humans.",
        "Parallel lines always meet at the edge of the universe.",
        "Euler's identity proves that 2 + 2 = 5.",
        "The golden ratio is only found in man-made structures.",
        "All odd numbers are prime numbers.",
        "Imaginary numbers are called imaginary because they don't exist.",
        "The Fibonacci sequence is the basis for all modern computing algorithms.",
        "Zero was first discovered in the 19th century.",
        "The concept of infinity was disproven by ancient Greek philosophers.",
        "All even numbers are divisible by 3.",
        "The sum of all natural numbers equals zero.",
        "Pi can be fully written out if you have a fast enough computer.",
        "Probability theory only works when predicting coin flips.",
        "Squares are just rectangles with better PR.",
    ]
    
    def combine_distractions():
        all_facts = (science_facts + historical_facts + daily_life + 
                    random_facts + philosophical_questions + mathematical_concepts + incorrect_facts)
        
        return random.choice(all_facts)

    sentences =  [combine_distractions() for _ in range(10)]
    paragraph = " ".join(sentences)
    if not paragraph.endswith((".", "?", "!")):
        paragraph += "."

    return paragraph