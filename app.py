"""
OptiSolve - AI-Powered OR Modeler
Uses Winston's Operations Research methodology.
Word problem -> Gemini (OR expert) -> structured LP -> PuLP solver -> LaTeX + Plotly results.
"""

import json
import re
from typing import Optional, Tuple, Union

import streamlit as st
import plotly.graph_objects as go
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, LpStatus, value

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="OptiSolve",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar: API key + branding (Winston's Operations Research)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📐 OptiSolve")
    st.markdown("*AI-Powered OR Modeler*")
    st.markdown("---")
    st.markdown(
        "**Developed by Efe G.**  \n"
        "*Industrial Engineering*  \n"
        "*Yaşar University*"
    )
    st.markdown("---")
    st.caption("Methodology: **Winston's Operations Research** (e.g. Diet, Work-Scheduling).")
    st.markdown("---")
    st.markdown("**API Key**")
    api_key = st.text_input(
        "GEMINI_API_KEY",
        type="password",
        placeholder="Enter your Gemini API key",
        label_visibility="collapsed",
        key="gemini_api_key",
    )
    if api_key:
        st.caption("✓ Key provided")
    else:
        st.caption("Required to extract LP from word problems.")

# ---------------------------------------------------------------------------
# Gemini: OR expert — extract structured LP from word problem
# ---------------------------------------------------------------------------
OR_EXPERT_SYSTEM = """You are an expert in Operations Research following Winston's textbook style.
Given a word problem (diet, scheduling, blending, resource allocation, etc.), extract a linear program in this exact JSON format—no other text, no markdown, only valid JSON:

{
  "sense": "Minimize",
  "variables": ["x1", "x2"],
  "objective_function": "0.6*x1 + 0.35*x2",
  "constraints": [
    {"lhs": "5*x1 + 7*x2", "op": ">=", "rhs": 8},
    {"lhs": "4*x1 + 2*x2", "op": ">=", "rhs": 15}
  ]
}

Rules:
- "sense" is exactly "Maximize" or "Minimize".
- "variables": list of variable names (e.g. x1, x2 or descriptive names like oats, corn). Use names that are valid in expressions (letters, numbers, underscore).
- "objective_function": single string, linear expression with * for multiplication (e.g. 20*x1 + 30*x2). No spaces required around *.
- "constraints": list of objects with "lhs" (linear expression string), "op" (one of ">=", "<=", "="), "rhs" (number).
- All variables are nonnegative unless the problem states otherwise; do not add nonnegativity constraints as separate rows.
- Output only the JSON object, no explanation."""

# Preferred model string; fallbacks if this one is not available (avoids 404 from wrong/versioned API).
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-pro"]


def get_available_gemini_model(api_key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Check which Gemini model is available. Returns (model_name_for_api, error_message).
    Uses list_models() so we do not call a broken or wrong API version.
    """
    if not api_key or not api_key.strip():
        return None, "API key is required."
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key.strip())
        available = set()
        for m in genai.list_models():
            name = getattr(m, "name", None) or getattr(m, "display_name", "") or ""
            if not name:
                continue
            available.add(name)
            if name.startswith("models/"):
                available.add(name.replace("models/", "", 1))
        if not available:
            return None, "No models returned from API. Check your API key and quota."
        preferred = [GEMINI_MODEL] + [x for x in GEMINI_FALLBACK_MODELS if x != GEMINI_MODEL]
        for candidate in preferred:
            if candidate in available:
                return candidate, None
            full = f"models/{candidate}"
            if full in available:
                return full, None
        pick = next((n for n in available if "flash" in n.lower() or "gemini" in n.lower()), None)
        if pick:
            return pick, None
        return None, "No supported Gemini model found. Available: " + ", ".join(sorted(available)[:10])
    except Exception as e:
        return None, f"Could not check models: {e}"


def extract_lp_with_gemini(word_problem: str, api_key: str) -> Tuple[Optional[dict], Optional[str]]:
    """Call Gemini to extract structured LP. Returns (parsed_dict, error_message)."""
    if not api_key or not word_problem.strip():
        return None, "API key and problem text are required."
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key.strip())
        model_name, err = get_available_gemini_model(api_key)
        if err or not model_name:
            return None, err or "Model not available."
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            OR_EXPERT_SYSTEM + "\n\nWord problem:\n\n" + word_problem.strip(),
            generation_config={"temperature": 0.1},
        )
        text = response.text.strip()
        # Remove optional markdown code fence
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        if not isinstance(data, dict) or "sense" not in data or "constraints" not in data:
            return None, "AI did not return a valid LP structure (sense, constraints)."
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON from AI: {e}"
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Parser: linear expression -> list of (coeff, var_name)
# ---------------------------------------------------------------------------
def parse_linear_term(s: str) -> tuple[float | None, str | None]:
    s = s.strip()
    if not s:
        return None, None
    m = re.match(r"^(-?\d*\.?\d*)\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*)$", s)
    if m:
        coeff, var = m.groups()
        return (float(coeff) if coeff else 1.0), var
    m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)$", s)
    if m:
        return 1.0, m.group(1)
    m = re.match(r"^(-?\d+\.?\d*)$", s)
    if m:
        return float(m.group(1)), None
    return None, None


def parse_linear_expression(text: str) -> list[tuple[float, str | None]]:
    text = text.strip().replace(" ", "")
    parts = re.split(r"\s*([+-])\s*", text)
    out = []
    sign = 1
    for p in parts:
        if p in ("+", "-"):
            sign = 1 if p == "+" else -1
            continue
        c, v = parse_linear_term(p)
        if c is not None:
            out.append((sign * c, v))
        sign = 1
    return out


def ai_lp_to_internal(ai: dict) -> Union[Tuple[type, list, list, list], Tuple[None, None, None, str]]:
    """Convert AI JSON to (sense, objective_terms, constraints_parsed, var_names). On error return (None,None,None,error_msg)."""
    try:
        sense_str = (ai.get("sense") or "").strip().lower()
        sense = LpMaximize if sense_str == "maximize" else LpMinimize
        obj_str = (ai.get("objective_function") or "").strip()
        constraints = ai.get("constraints") or []
        var_names = list(ai.get("variables") or [])
        if not obj_str:
            return None, None, None, "Missing objective_function."
        obj_terms = parse_linear_expression(obj_str)
        if not var_names:
            seen = set()
            for c, v in obj_terms:
                if v:
                    seen.add(v)
            for c in constraints:
                lhs = (c.get("lhs") or "").strip()
                for co, v in parse_linear_expression(lhs):
                    if v:
                        seen.add(v)
            var_names = sorted(seen)
        if not var_names:
            return None, None, None, "No decision variables found."
        constraints_parsed = []
        for c in constraints:
            lhs = (c.get("lhs") or "").strip()
            op = (c.get("op") or "==").strip().replace("==", "=")
            if op not in (">=", "<=", "="):
                continue
            try:
                rhs = float(c.get("rhs", 0))
            except (TypeError, ValueError):
                continue
            terms = parse_linear_expression(lhs)
            constraints_parsed.append((terms, op, rhs))
        return sense, obj_terms, constraints_parsed, var_names
    except Exception as e:
        return None, None, None, str(e)


# ---------------------------------------------------------------------------
# PuLP solver (Winston-compatible)
# ---------------------------------------------------------------------------
def build_and_solve_lp(sense, objective_terms, constraints_parsed, var_names):
    """Build PuLP model and solve. Returns (status, obj_value, var_values, error_msg)."""
    try:
        prob = LpProblem("OptiSolve_LP", sense)
        vars_dict = {v: LpVariable(v, lowBound=0) for v in var_names}
        obj_expr = 0
        for c, v in objective_terms:
            if v and v in vars_dict:
                obj_expr += c * vars_dict[v]
            elif v is None:
                obj_expr += c
        prob += obj_expr
        for terms, op, rhs in constraints_parsed:
            expr = 0
            for c, v in terms:
                if v and v in vars_dict:
                    expr += c * vars_dict[v]
                elif v is None:
                    expr += c
            if op == "<=":
                prob += expr <= rhs
            elif op == ">=":
                prob += expr >= rhs
            else:
                prob += expr == rhs
        prob.solve()
        status = LpStatus[prob.status]
        if status != "Optimal":
            return status, None, None, f"Solver status: {status}"
        obj_value = value(prob.objective)
        var_values = {v: value(vars_dict[v]) for v in var_names}
        return status, obj_value, var_values, None
    except Exception as e:
        return None, None, None, str(e)


# ---------------------------------------------------------------------------
# LaTeX: format objective and constraints for display (Z = ... and subscripts)
# ---------------------------------------------------------------------------
def to_latex_var(name: str) -> str:
    """Turn x1 -> x_1 for LaTeX."""
    if not name or not name[-1].isdigit():
        return name
    i = next((i for i, c in enumerate(name) if c.isdigit()), len(name))
    base, num = name[:i], name[i:]
    return f"{base}_{{{num}}}" if base else name


def objective_to_latex(sense: type, obj_terms: list) -> str:
    parts = []
    for c, v in obj_terms:
        if v is None:
            parts.append(str(c))
        else:
            cv = to_latex_var(v)
            if c == 1:
                parts.append(cv)
            elif c == -1:
                parts.append("-" + cv)
            else:
                parts.append(f"{c} {cv}")
    expr = " + ".join(parts).replace("+ -", "- ")
    sense_str = "Maximize" if sense == LpMaximize else "Minimize"
    return f"{sense_str} \\quad Z = {expr}"


def constraint_to_latex(terms: list, op: str, rhs: float) -> str:
    parts = []
    for c, v in terms:
        if v is None:
            parts.append(str(c))
        else:
            cv = to_latex_var(v)
            if c == 1:
                parts.append(cv)
            elif c == -1:
                parts.append("-" + cv)
            else:
                parts.append(f"{c} {cv}")
    lhs = " + ".join(parts).replace("+ -", "- ")
    op_latex = "\\leq" if op == "<=" else ("\\geq" if op == ">=" else "=")
    return f"{lhs} \\quad {op_latex} \\quad {rhs}"


# ---------------------------------------------------------------------------
# Main UI: word problem text area -> Solve -> LaTeX + Plotly
# ---------------------------------------------------------------------------
st.title("📐 OptiSolve")
st.markdown("**AI-Powered OR Modeler** — Paste a word problem; the AI extracts the linear program (Winston-style), then we solve it with PuLP.")

st.header("1. Word problem")
word_problem = st.text_area(
    "Paste your optimization word problem here (e.g. diet, scheduling, blending).",
    height=280,
    placeholder="e.g. A dietitian wants to minimize the cost of a diet. Food 1 costs $0.60 per unit and provides 5 units of starch, 4 of protein, 2 of vitamins. Food 2 costs $0.35 per unit and provides 7, 2, 1 units respectively. Daily requirements are at least 8, 15, and 3 units. How much of each food should be used?",
    label_visibility="collapsed",
    key="word_problem",
)

col1, col2, _ = st.columns([1, 1, 3])
with col1:
    solve_clicked = st.button("Solve", type="primary", use_container_width=True)

st.header("2. Mathematical formulation (extracted by AI)")

if solve_clicked and word_problem.strip():
    if not api_key:
        st.error("Please enter your GEMINI_API_KEY in the sidebar.")
    else:
        with st.spinner("Asking Gemini (OR expert) to extract the LP…"):
            ai_dict, err = extract_lp_with_gemini(word_problem, api_key)
        if err:
            st.error(err)
        else:
            sense, obj_terms, constraints_parsed, var_names = ai_lp_to_internal(ai_dict)
            if sense is None:
                st.error(var_names)  # error message
            else:
                with st.expander("Decision variables", expanded=True):
                    st.write(", ".join(f"**{v}**" for v in var_names))
                with st.expander("Objective function", expanded=True):
                    st.latex(objective_to_latex(sense, obj_terms))
                with st.expander("Constraints", expanded=True):
                    for terms, op, rhs in constraints_parsed:
                        st.latex(constraint_to_latex(terms, op, rhs))

                status, obj_value, var_values, err = build_and_solve_lp(
                    sense, obj_terms, constraints_parsed, var_names
                )
                if err:
                    st.error(err)
                else:
                    st.header("3. Optimal results")
                    st.success(f"**Status:** {status}")
                    st.metric("Objective value (Z)", f"{obj_value:.4f}")
                    st.subheader("Optimal variable values")
                    for v in var_names:
                        st.latex(f"{to_latex_var(v)}^* = {var_values[v]:.4f}")
                    st.subheader("Solution chart")
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=list(var_values.keys()),
                                y=list(var_values.values()),
                                marker_color="#1E3A5F",
                                text=[f"{var_values[k]:.4f}" for k in var_values],
                                textposition="outside",
                            )
                        ]
                    )
                    fig.update_layout(
                        title="Optimal decision variable values",
                        xaxis_title="Variable",
                        yaxis_title="Value",
                        template="plotly_white",
                        height=400,
                        margin=dict(t=50, b=50),
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif solve_clicked:
    st.warning("Please enter a word problem in the text area above.")

else:
    with st.expander("Decision variables", expanded=True):
        st.caption("Decision variables will appear here after you run **Solve**.")
    with st.expander("Objective function", expanded=True):
        st.caption("Objective function will appear here (e.g. \\( \\min Z = \\ldots \\)).")
    with st.expander("Constraints", expanded=True):
        st.caption("Constraints will appear here.")
    st.header("3. Optimal results")
    st.info("Click **Solve** to extract the LP with Gemini and see optimal results and chart here.")
