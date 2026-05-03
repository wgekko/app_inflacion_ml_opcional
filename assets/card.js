const flashcards = [
    {
        question: "Modelo SARIMA-RED LSTM unistep",
        hint: "Modelo#1",
        answer: "SARIMA + LSTM que captura patrones lineales, no lineales en series temporales, realiza predicciones unistep con mayor precisión combinando estadística clásica y deep learning."
    },
    {
        question: "Modelo SARIMA HOLT-WINTERS",
        hint: "Modelo#2",
        answer: "Modelo que combina SARIMA y Holt-Winters para capturar tendencia, estacionalidad y patrones temporales. Mejora el pronóstico al integrar enfoques estadísticos clásicos complementarios."
    },
    {
        question: "Proyecciones de IA y Econometría",
        hint: "Modelo#3",
        answer: "Proyecciones a 6 meses que combinan IA y econometría para anticipar tendencias y mejorar la toma de decisiones."
    },
    {
        question: "Modelos GRU-TCN-TFT",
        hint: "Modelo#4",
        answer: "Modelo híbrido GRU–TCN–Temporal Fusion Transformer para predecir inflación capturando dependencias temporales complejas."
    }
];

function initFlashcards(cards) {
    let currentIndex = 0;
    let isAnimating = false;
    const container = document.getElementById("card-container");
    const nextBtn = document.getElementById("next-btn");
    const prevBtn = document.getElementById("prev-btn");

    const createCard = ({ question, hint, answer }) => {
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `
            <h1>${question}</h1>
            <p>${hint}</p>
            <p class="answer">${answer}</p>
        `;
        card.querySelector(".answer").addEventListener("click", (e) => {
            e.target.classList.add("revealed");
        });
        return card;
    };

    const showCard = (index, direction) => {
        if (isAnimating) return;
        isAnimating = true;
        const newCard = createCard(cards[index]);
        if (direction === "prev") newCard.classList.add("enter-left");
        container.appendChild(newCard);

        requestAnimationFrame(() => {
            newCard.classList.add("show");
            if (direction === "prev") newCard.classList.remove("enter-left");
        });

        const oldCard = container.children.length > 1 ? container.firstChild : null;
        if (oldCard) {
            oldCard.classList.remove("show");
            oldCard.classList.add(direction === "next" ? "exit-left" : "exit-right");
            setTimeout(() => { oldCard.remove(); isAnimating = false; }, 1000);
        } else {
            setTimeout(() => { isAnimating = false; }, 1000);
        }
    };

    nextBtn.addEventListener("click", () => {
        currentIndex = (currentIndex + 1) % cards.length;
        showCard(currentIndex, "next");
    });

    prevBtn.addEventListener("click", () => {
        currentIndex = (currentIndex - 1 + cards.length) % cards.length;
        showCard(currentIndex, "prev");
    });

    showCard(currentIndex, "next");
}

initFlashcards(flashcards);