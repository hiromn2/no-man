# Knowledge Base Changelog

All changes to `knowledge/seed_graph.json`, `knowledge/nodes.json`, and
`knowledge/governance.json` must be recorded here.

**Policy: Append only — never rewrite history.**

---

## 2025-01-15: Initial seed graph created from literature

**Action:** Initial creation of the No-Man knowledge base  
**Files affected:**
- `knowledge/nodes.json` — created (28 nodes)
- `knowledge/seed_graph.json` — created (41 edges)
- `knowledge/governance.json` — created (promotion and immutability rules)

**Node count:** 28  
**Edge count:** 41  
**Edge ID range:** E-001 through E-041  
**All edge status:** tested  

**Node categories:**
- Financial (F1–F8): bank_loan_volume, bank_capital_ratio, bank_net_interest_margin, bank_npl_ratio, bank_fee_income, bank_operating_cost_ratio, local_deposit_base, bank_securities_portfolio
- Real Economy (R1–R6): regional_sme_credit_access, regional_employment, regional_gdp, regional_sme_investment, regional_housing_market, local_government_fiscal_health
- Institutional (I1–I4): main_bank_relationship_trust, bank_branch_network, regional_credit_market_competition, bank_governance_quality
- Behavioral (B1–B4): borrower_confidence, depositor_confidence, regional_entrepreneur_activity, bank_reputation_regional
- Regulatory (Reg1–Reg4): fsa_supervisory_scrutiny, bank_regulatory_capital_pressure, basel_compliance_burden, fsa_prompt_corrective_action_risk
- Macro (M1–M2): regional_population_trend, boj_policy_rate

**Sources used:**
- Uchida, H., Udell, G. F., & Yamori, N. (2008). Loan officers and relationship lending. — Japanese SME lending, relationship banking
- Hoshi, T., & Kashyap, A. K. (2001). Corporate Financing and Governance in Japan. — Japanese banking crisis mechanisms
- Watanabe, W. (2007). Prudential regulation and the "credit crunch": Evidence from Japan. — Japanese banking regulation and NPL effects
- Peek, J., & Rosengren, E. S. (2000). Collateral damage: Effects of the Japanese bank crisis on real activity in the United States. — NPL-capital-lending channel
- Hosono, K. (2006). The transmission mechanism of monetary policy in Japan: Evidence from banks' balance sheets. — Capital ratio and loan supply
- Miyajima, H., & Kuroki, F. (2007). The unwinding of cross-shareholding in Japan. — Bank securities portfolio and capital
- Ogawa, K., & Suzuki, K. (1998). Uncertainty and investment: some evidence from the panel data of Japanese manufacturing firms. — Credit and investment link
- Tsuruta, D. (2014). Bank loan availability and trade credit for small businesses during the financial crisis. — SME credit and employment
- Ogura, Y. (2010). Interbank competition and information production. — Regional banking competition and NIM
- Bank of Japan regional economic reports (各種年度) — Population, deposit base, employment, and local government fiscal data for Japanese regions

**Confidence assignment rationale:** Conservative assignments used throughout. "high" confidence assigned only where the source directly studies Japanese regional banks or very close analogs (Japanese banking crisis, Japanese SMEs, Japan panel data). "medium" assigned where Japan evidence exists but is indirect. "low" assigned where the relationship is inferred from theory or international studies.

**Graph structure notes:**
- All 41 edges confirmed as forming a directed acyclic graph (DAG); validated by topological sort
- Cycle risks noted at edge level where real-world feedback loops may exist outside the DAG structure
- 5 edges marked non-monotone (bank_securities_portfolio → bank_capital_ratio; boj_policy_rate → bank_net_interest_margin)
- 4 edges marked as research gaps with suggested empirical tests
- 2 edges with regime conditions (interest rate normalization regime)

**Human review status:** PENDING — Initial seed graph requires verification by domain expert (Japanese regional banking specialist) before use in production advisory reports.

**Signed by:** Project initialization (automated build from literature)

---

*[Future entries should be appended below this line]*
