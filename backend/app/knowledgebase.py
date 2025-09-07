from typing import List,Tuple,Optional

class KnowledgeBase:


    def __init__(self):
        self.entries: List[Tuple[List[str], str]] = [
            ( ["emergency", "urgent", "severe bleeding", "trouble breathing"],
            "If this is an emergency (e.g., severe pain, trouble breathing, heavy bleeding), call your local emergency number or visit the nearest hospital immediately.") ,
            ( ["screening", "mammogram", "age"],
            "Breast cancer screening with mammography is commonly recommended starting at age 40–50 depending on guidelines; consult a clinician for a plan based on your risk."),
            ( ["hotline", "helpline"],
            "For counseling and cancer support services, contact your national cancer society helpline (availability varies by country)."),
            ( ["disclaimer"],
            "This chatbot does not provide medical diagnosis. Always consult a qualified healthcare professional for personal medical advice."),
            ]


    def maybe_answer(self, question: str) -> Optional[str]:
        q = question.lower()
        for keywords, reply in self.entries:
            if all(k in q for k in keywords):
                return reply
        return None