document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("user-input");
  const output = document.getElementById("output");

  input.addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
      const query = input.value.trim();
      if (!query) return;

      output.innerHTML = "<p>Processing...</p>";

      fetch("/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query }),
      })
        .then((res) => res.json())
        .then((data) => {
          output.innerHTML = `<p>${data.response}</p>`;
        })
        .catch((err) => {
          output.innerHTML = `<p>Error: ${err.message}</p>`;
        });

      input.value = "";
    }
  });
});
