"""Generate Unisoft Human Resources Policy.pdf for RAG."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

SECTIONS = [
    (
        "1. Purpose and scope",
        "This Human Resources Policy applies to all employees of Unisoft. It explains leave, "
        "working hours, conduct, hiring, and workplace safety. This document is internal guidance "
        "for employees and managers. It is not a contract of employment.",
    ),
    (
        "2. Working hours",
        "Standard Unisoft working hours are Monday to Friday, 9:00 to 18:00, including a one-hour "
        "unpaid lunch break. Core hours are 10:00 to 16:00. Employees may request flexible start "
        "times between 8:00 and 10:00 with manager approval. Saturday and Sunday are rest days "
        "unless a written roster says otherwise. Overtime must be approved in writing before it is worked.",
    ),
    (
        "3. Annual leave",
        "Unisoft annual leave entitlement is 18 working days per calendar year for employees with "
        "fewer than five years of continuous service. Employees with five or more years of continuous "
        "service receive 21 working days per calendar year. Annual leave is counted in working days "
        "(Monday to Friday) and does not include public holidays. Leave of more than three consecutive "
        "working days should be requested at least ten working days in advance through the HR assistant "
        "or the HR portal. Unused annual leave up to five days may be carried into the next calendar year; "
        "any remaining unused days expire on 31 March of the following year.",
    ),
    (
        "4. Sick leave",
        "Unisoft provides up to 14 days of paid sick leave per calendar year with a medical certificate "
        "from a registered doctor. The first two days of any sick-leave episode may be self-certified. "
        "Employees must notify their manager before 10:00 on the first day of absence. Hospitalisation "
        "leave of up to 60 days per calendar year may be granted with hospital documents.",
    ),
    (
        "5. Other leave",
        "Compassionate leave is three working days for the death of an immediate family member. "
        "Marriage leave is three working days, taken within 30 days of the wedding. Maternity leave "
        "and paternity leave follow applicable local employment law. Study leave of up to five working "
        "days per year may be approved for job-related training with HR sign-off.",
    ),
    (
        "6. Public holidays",
        "Unisoft observes official public holidays of the country where the employee is employed. "
        "If a public holiday falls on a weekend, a substitute weekday off is given only when local law "
        "or a written Unisoft notice requires it.",
    ),
    (
        "7. Remote work",
        "Eligible employees may work remotely up to two days per week after completing probation, "
        "subject to manager approval. Remote days must be recorded in the HR portal. Unisoft equipment "
        "must be used for company systems. Employees must be reachable on company chat during core hours.",
    ),
    (
        "8. Probation and notice",
        "New hires serve a three-month probation. During probation, either party may end employment "
        "with seven calendar days' written notice. After probation, the standard notice period is 30 "
        "calendar days unless the employment contract states a longer period.",
    ),
    (
        "9. Code of conduct",
        "Employees must treat colleagues and clients with respect. Harassment, discrimination, and "
        "violence are prohibited. Confidential client and employee data must not be shared outside "
        "Unisoft without authorisation. Conflicts of interest must be declared to HR in writing. "
        "Gifts from vendors above a value of 50 currency units must be declined or reported to HR.",
    ),
    (
        "10. Performance",
        "Unisoft runs a formal performance review twice each year, in June and December. Goals are "
        "agreed with the manager in January. A rating of Needs Improvement requires a 60-day performance "
        "plan. Promotion and bonus decisions consider review ratings, business need, and budget.",
    ),
    (
        "11. How to ask HR",
        "Questions about this policy can be sent to the Unisoft HR assistant chatbot or to hr@unisoft.example. "
        "Holiday applications must include user name, email, purpose, start date, and end date. HR will "
        "count working days Monday to Friday and send a confirmation email after the employee confirms "
        "the details are correct.",
    ),
]


def main() -> None:
    targets = [
        Path("/workspace/Unisoft Human Resources Policy.pdf"),
        Path("/workspace/tool-test/Unisoft Human Resources Policy.pdf"),
    ]
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontSize=16,
        spaceAfter=12,
    )
    heading = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=0.9 * inch,
            rightMargin=0.9 * inch,
            topMargin=0.8 * inch,
            bottomMargin=0.8 * inch,
        )
        story = [
            Paragraph("Unisoft Human Resources Policy", title),
            Paragraph("Effective date: 1 January 2026. Version 1.0.", body),
            Spacer(1, 8),
        ]
        for heading_text, para in SECTIONS:
            story.append(Paragraph(heading_text, heading))
            story.append(Paragraph(para, body))
        doc.build(story)
        print("wrote", path)


if __name__ == "__main__":
    main()
