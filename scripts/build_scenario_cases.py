"""Freeze an authored diagnostic corpus before observing any new model outputs.

The corpus (examples/audit-003-scenarios.jsonl) is in Portuguese by design: every Portuguese string below is
benchmark data and must stay byte-identical so the frozen SHA-256 still matches.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    cases = []
    def add(domain, texts, labels, instructions, kind="rubric"):
        assert len(texts) == 18
        for i, item in enumerate(texts):
            state, label, rationale = item
            cases.append({"id": f"{domain}-{i:02d}", "domain": domain,
                          "split": "development" if i < 3 else "calibration" if i < 9 else "test",
                          "state": state, "question": {"type":"choice", "instructions":instructions,
                          "criteria":labels}, "target": {k:float(k == label) for k in labels},
                          "target_kind":kind, "rationale":rationale})

    add("deterministic", [
        ("A caixa azul está na sala 2. A verde está na sala 5.","yes","Local explícito."),
        ("A caixa azul está na sala 8, não na sala 2.","no","Negação explícita."),
        ("A caixa verde está na sala 2. Nada consta sobre a azul.","unknown","Referente diferente."),
        ("O inventário registra: caixa azul, sala 2.","yes","Registro direto."),
        ("A azul foi retirada da sala 2 e colocada na sala 4.","no","Estado mais recente."),
        ("A caixa azul saiu do depósito. Seu destino não foi registrado.","unknown","Destino ausente."),
        ("A azul estava na sala 4, mas agora está na sala 2.","yes","Estado atual."),
        ("Há apenas uma caixa azul e ela está na sala 7.","no","Local incompatível."),
        ("Talvez a caixa azul esteja na sala 2; ninguém verificou.","unknown","Possibilidade não confirma."),
        ("Relatório confirmado hoje: a caixa azul permanece na sala 2.","yes","Confirmação atual."),
        ("A sala 2 está vazia. A caixa azul está no corredor.","no","Sala vazia e outro local."),
        ("A caixa azul estava na sala 2 ontem. Hoje ela foi transportada para um local desconhecido.","unknown","Histórico não determina presente."),
        ("Blue box location: room 2. This record is current and verified.","yes","Explicit verified record."),
        ("The blue box is in room 3. Room 2 contains only a red box.","no","Different box."),
        ("The label reads 'blue box'; the location field is blank.","unknown","Missing location."),
        ("Somente as caixas azul e preta estão atualmente na sala 2.","yes","Membro do conjunto."),
        ("A caixa azul não está em nenhuma sala; está no caminhão.","no","Exclusão explícita."),
        ("Uma caixa está na sala 2, mas sua cor não foi anotada.","unknown","Cor ausente."),
    ], {"yes":"Sim, confirmado", "no":"Não, contradito", "unknown":"Não é possível determinar"},
    "A caixa azul está atualmente na sala 2? Use apenas os fatos do texto; informação ausente não equivale a negação.", "factual")

    add("sentence", [
        ("O trem chegou às oito.","statement","Declaração."),
        ("A que horas o trem chegou?","question","Pedido de informação."),
        ("Por favor, feche a janela.","request","Pedido de ação."),
        ("Não houve reunião nesta terça-feira.","statement","Declaração negativa."),
        ("Quem deixou as luzes acesas?","question","Interrogação informativa."),
        ("Envie o recibo até amanhã.","request","Imperativo."),
        ("Segundo a ata, a proposta foi aprovada.","statement","Relato declarativo."),
        ("Where is the nearest station?","question","Information request."),
        ("Please save the file before closing the application.","request","Action request."),
        ("Ela perguntou onde estava a chave.","statement","Relata uma pergunta; não pergunta ao leitor."),
        ("Ainda há lugares disponíveis?","question","Pergunta sobre disponibilidade."),
        ("Você poderia baixar o volume, por favor?","request","Forma interrogativa, função de pedido."),
        ("A placa contém a frase 'Você chegou?'.","statement","Citação dentro de declaração."),
        ("Qual das duas estradas leva ao museu?","question","Busca informação."),
        ("Não compartilhe esta senha com ninguém.","request","Pedido negativo."),
        ("I wonder whether it will rain tomorrow.","statement","Expresses a thought rather than directly asking the reader."),
        ("How many pages does the report contain?","question","Information request."),
        ("Could you send me the report, please?","request","Indirect request for action."),
    ], {"statement":"Declaração ou relato", "question":"Pergunta que busca informação", "request":"Pedido para realizar ou evitar uma ação"},
    "Classifique a função comunicativa principal da frase inteira. Um pedido de ação pode ter forma interrogativa; uma pergunta citada não transforma um relato em pergunta.", "linguistic_rubric")

    add("sentiment", [
        ("Adorei o atendimento, resolveram tudo com muita atenção!","positive","Elogio explícito."),
        ("O pacote tem três peças e pesa dois quilos.","neutral","Descrição sem avaliação."),
        ("Foi horrível: demorou e ainda veio quebrado.","negative","Crítica explícita."),
        ("O filme me encantou do início ao fim.","positive","Avaliação favorável."),
        ("A sessão começa às 19h e dura noventa minutos.","neutral","Informação factual."),
        ("Não gostei do filme; o roteiro é confuso.","negative","Insatisfação direta."),
        ("Não esperava muito, mas o jantar foi excelente.","positive","Desfecho favorável."),
        ("O jantar foi servido no salão do primeiro andar.","neutral","Local, sem avaliação."),
        ("Great, another update that deletes my settings. Just what I needed.","negative","Sarcasmo e consequência negativa."),
        ("O preço é alto, mas a qualidade compensou e estou muito satisfeito.","positive","Avaliação global explicitamente favorável."),
        ("O preço passou de 20 para 22 euros nesta semana.","neutral","Mudança factual sem atitude expressa."),
        ("A embalagem é bonita, mas o produto falhou e me arrependi da compra.","negative","Conclusão desfavorável."),
        ("I wasn't disappointed at all; the concert was wonderful.","positive","Negated disappointment plus praise."),
        ("The concert took place on Saturday in the municipal hall.","neutral","Factual description."),
        ("Maravilha, fiquei duas horas esperando para descobrir que cancelaram tudo.","negative","Sarcasmo em experiência ruim."),
        ("Demorou a chegar, porém superou minhas expectativas. Recomendo.","positive","Recomendação final."),
        ("A entrega está registrada para quinta-feira, entre 9h e 12h.","neutral","Agendamento sem opinião."),
        ("The interface looks nice, but losing my work twice made this unusable.","negative","Overall negative assessment."),
    ], {"positive":"Positivo", "neutral":"Neutro, sem avaliação afetiva clara", "negative":"Negativo"},
    "Qual é o sentimento global expresso pelo autor sobre a experiência ou objeto descrito? Considere negação, contraste e sarcasmo; não classifique apenas palavras isoladas.")

    add("subjective_tone", [
        ("Obrigado pela ajuda! Quando puder, poderia revisar este ponto comigo?","warm","Agradecimento e pedido cortês."),
        ("O documento deve ser revisado até sexta-feira.","neutral","Instrução impessoal sem ataque."),
        ("Você não entende nada e seu trabalho é uma vergonha.","hostile","Insulto pessoal."),
        ("Entendo a dificuldade. Vamos resolver juntos; agradeço seu esforço.","warm","Empatia e apoio."),
        ("A solicitação foi registrada sob o número 42.","neutral","Registro administrativo."),
        ("Pare de fazer perguntas idiotas e aprenda a trabalhar.","hostile","Desqualificação pessoal."),
        ("Discordo da proposta, mas valorizo seu trabalho e gostaria de conversar.","warm","Discordância respeitosa com valorização."),
        ("Please submit the signed form by 5 p.m.","neutral","Routine polite instruction, no personal warmth."),
        ("Only an incompetent person would make a mistake like yours.","hostile","Personal insult."),
        ("Sei que foi uma semana difícil. Obrigado por tentar; posso ajudar amanhã.","warm","Reconhecimento e oferta de ajuda."),
        ("Não concordo com o cálculo. O total correto é 18.","neutral","Correção sem insulto."),
        ("Até uma criança faria isso melhor que você.","hostile","Humilhação comparativa."),
        ("Thanks for flagging this. I appreciate your care and I'm happy to help.","warm","Gratitude and support."),
        ("Your request cannot be approved because the required field is missing.","neutral","Negative decision without personal attack."),
        ("Congratulations on proving once again how useless you are.","hostile","Sarcastic insult."),
        ("Sinto muito pelo transtorno. Quero ouvir você e encontrar uma solução juntos.","warm","Empatia e colaboração."),
        ("A versão enviada contém dois erros; corrija as linhas 4 e 8.","neutral","Crítica técnica sem ataque pessoal."),
        ("Não perca meu tempo, seu incompetente.","hostile","Insulto direto."),
    ], {"warm":"Acolhedor: empatia, apoio ou valorização pessoal", "neutral":"Neutro/profissional: instrução ou crítica sem ataque pessoal nem acolhimento explícito", "hostile":"Hostil: insulto, humilhação ou ataque pessoal"},
    "Avalie o tom interpessoal usando a rubrica fornecida. Uma recusa ou crítica técnica não é hostilidade por si só. Esta é uma interpretação de tom, não uma verdade objetiva sobre a intenção do autor.")

    add("robotic_style", [
        ("Sua solicitação foi processada com sucesso. Agradecemos o contato. Permanecemos à disposição para quaisquer esclarecimentos adicionais.","formulaic","Sequência de fórmulas administrativas genéricas."),
        ("nossa, eu jurava que tinha salvo isso ontem... pera, achei na outra pasta haha","conversational","Hesitação, autocorreção e informalidade."),
        ("Certo.","insufficient","Texto curto sem evidência suficiente."),
        ("Prezado usuário, informamos que sua demanda foi encaminhada ao setor responsável. Solicitamos que aguarde o prazo estipulado.","formulaic","Molde administrativo impessoal."),
        ("Cara, o café caiu bem em cima da minha anotação. Agora o mapa parece um polvo.","conversational","Detalhe concreto e expressão espontânea."),
        ("A reunião começa às nove.","insufficient","Uma frase factual não determina estilo."),
        ("Em primeiro lugar, é importante ressaltar a relevância do tema. Em segundo lugar, cabe destacar seus múltiplos aspectos. Em conclusão, trata-se de uma questão relevante.","formulaic","Estrutura repetitiva com pouco conteúdo específico."),
        ("I thought I'd hate it, but then that tiny dog stole my sandwich and somehow made my day.","conversational","Specific anecdote and personal voice."),
        ("Thank you.","insufficient","Too little stylistic evidence."),
        ("Identificamos sua manifestação. Reiteramos nosso compromisso com a excelência. Sua satisfação é nossa prioridade. Agradecemos sua compreensão.","formulaic","Fórmulas institucionais intercambiáveis."),
        ("Tá, fui olhar de novo e você tinha razão. Eu tava lendo a coluna errada, que vergonha.","conversational","Autocorreção coloquial contextual."),
        ("O relatório contém doze páginas.","insufficient","Fato isolado."),
        ("We acknowledge receipt of your inquiry. Your feedback is important to us. We remain committed to providing optimal solutions tailored to your needs.","formulaic","Generic institutional template."),
        ("Wait, did I leave the keys in the fridge? Yep. Don't ask. It's been that kind of morning.","conversational","Personal, fragmented anecdote."),
        ("The file is attached.","insufficient","Short factual message."),
        ("Para garantir uma experiência satisfatória, recomendamos seguir os procedimentos estabelecidos. Caso necessite de assistência adicional, entre em contato pelos canais oficiais.","formulaic","Padronização e generalidade."),
        ("Eu ia escrever um textão, mas olha: gostei. Gostei mesmo. Só troca aquela música do elevador, pelo amor.","conversational","Ritmo coloquial e detalhe específico."),
        ("Sim, recebi ontem.","insufficient","Pouca evidência estilística."),
    ], {"formulaic":"Estilo padronizado/robótico: fórmulas genéricas, impessoalidade e repetição", "conversational":"Estilo conversacional: voz pessoal, detalhes específicos e espontaneidade aparente", "insufficient":"Texto insuficiente para avaliar o estilo"},
    "Classifique apenas o estilo aparente do texto. Não infira se o autor é uma IA ou uma pessoa: ambos podem escrever em qualquer desses estilos. Use 'insufficient' quando a evidência estilística for insuficiente.")

    colors = {"red":"Vermelha / red", "blue":"Azul / blue", "green":"Verde / green"}
    counts = [(1,1,1),(6,3,1),(1,2,7),(2,2,2),(1,8,1),(7,2,1),
              (0,3,3),(2,0,8),(4,6,0),(5,5,5),(2,5,3),(8,1,1),
              (0,0,10),(0,10,0),(10,0,0),(1,4,5),(3,3,4),(9,0,1)]
    for i,(r,b,g) in enumerate(counts):
        total=r+b+g
        cases.append({"id":f"random-{i:02d}","domain":"random",
            "split":"development" if i<3 else "calibration" if i<9 else "test",
            "state":f"Uma urna contém {r} bolas vermelhas, {b} azuis e {g} verdes. As bolas são idênticas exceto pela cor. Uma bola será sorteada uniformemente ao acaso, sem observação antecipada.",
            "question":{"type":"choice","instructions":"Qual será a cor da próxima bola sorteada? Represente a incerteza do sorteio nas probabilidades das alternativas, usando somente a composição da urna.","criteria":colors},
            "target":dict(zip(colors,[r/total,b/total,g/total])), "target_kind":"known_distribution",
            "rationale":"Probabilidades exatas por contagem. Não há um resultado sorteado observado; acerto empírico não é definido."})
    assert len(cases)==108 and len({c['state'] for c in cases})==108
    for c in cases:
        assert abs(sum(c['target'].values())-1)<1e-9
    return cases


if __name__ == '__main__':
    path=ROOT/'examples/audit-003-scenarios.jsonl'
    data=''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in build()).encode('utf-8')
    with path.open('xb') as f:f.write(data)
    print(json.dumps({'cases':108,'sha256':hashlib.sha256(data).hexdigest()}))
