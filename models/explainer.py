def format_int_ptbr(value: int) -> str:
    """
    Formata inteiros no padrão brasileiro:
    1000 -> 1.000
    1000000 -> 1.000.000
    """
    return f"{value:,}".replace(",", ".")


def generate_business_explanation(results: dict) -> dict:
    """
    Gera explicações em linguagem de negócio a partir dos resultados do modelo.
    Retorna blocos de texto prontos para exibição no frontend.
    """

    meta = results["meta"]
    rf = results["rf"]
    et = results["et"]

    n = meta["n_samples"]
    acc_rf = rf["test_accuracy"]
    acc_et = et["test_accuracy"]

    # distribuição real
    cm = rf["confusion_matrix"]
    total_accept = cm[1][0] + cm[1][1]
    total_reject = cm[0][0] + cm[0][1]

    pct_accept = total_accept / (total_accept + total_reject)
    pct_reject = 1 - pct_accept

    # feature importance
    importances = rf["feature_importance"]
    axes = [meta["x_axis"], meta["y_axis"]]
    ranked = sorted(zip(axes, importances), key=lambda x: x[1], reverse=True)

    main_feature, main_weight = ranked[0]
    second_feature, second_weight = ranked[1]

    # -------------------------
    # TEXTO 1 — RESUMO EXECUTIVO
    # -------------------------
    summary = (
        f"Nesta análise, o sistema avaliou {format_int_ptbr(n)} situações reais de oferta de cupons. "
        f"Com base nos padrões identificados, aproximadamente "
        f"{pct_accept:.0%} dos cupons tendem a ser aceitos, enquanto "
        f"{pct_reject:.0%} apresentam maior chance de rejeição. "
        "Isso indica um cenário de aceitação moderada, com potencial de otimização."
    )

    # -------------------------
    # TEXTO 2 — COMO O SISTEMA DECIDIU
    # -------------------------
    model_logic = (
        "O sistema aprendeu a diferenciar situações favoráveis e desfavoráveis "
        "comparando combinações de características do usuário e do contexto da oferta. "
        "Essas combinações permitem estimar, com boa confiabilidade, "
        "quando um cupom tem maior chance de ser aceito."
    )

    # -------------------------
    # TEXTO 3 — FATORES DECISIVOS
    # -------------------------
    features_text = (
        f"O fator mais relevante nesta análise foi '{main_feature}', "
        f"sendo responsável por aproximadamente {main_weight:.0%} da decisão do sistema. "
        f"Em seguida, '{second_feature}' também apresentou influência significativa, "
        f"contribuindo com cerca de {second_weight:.0%}. "
        "Esses fatores são determinantes para direcionar melhor a oferta de cupons."
    )

    # -------------------------
    # TEXTO 4 — CONFIABILIDADE
    # -------------------------
    confidence = (
        f"O desempenho do sistema foi consistente, com taxa de acerto próxima de "
        f"{acc_rf:.0%}. Isso significa que, na maioria dos casos, "
        "as decisões tomadas pelo sistema refletem corretamente "
        "o comportamento observado nos dados históricos."
    )

    # -------------------------
    # TEXTO 5 — CONCLUSÃO DE NEGÓCIO
    # -------------------------
    conclusion = (
        "Na prática, isso indica que o sistema pode ser utilizado como apoio "
        "na tomada de decisão para ofertas de cupons, ajudando a reduzir desperdícios "
        "e aumentar a taxa de aceitação. Ajustes nos filtros ou no público-alvo "
        "podem elevar ainda mais os resultados obtidos."
    )

    return {
        "summary": summary,
        "model_logic": model_logic,
        "key_factors": features_text,
        "confidence": confidence,
        "conclusion": conclusion,
    }
