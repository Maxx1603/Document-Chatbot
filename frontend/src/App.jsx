import { useState } from "react";
import axios from "axios";

function App() {

  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  // ---------------- UPLOAD PDF ----------------
  const uploadPDF = async () => {

    if (!file) {
      alert("Please select a PDF");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {

      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      console.log(response.data);

      alert("PDF Uploaded Successfully");

    } catch (error) {

      console.log(error);

      alert(
        error.response?.data?.error ||
        "Upload Failed"
      );

    } finally {

      setLoading(false);

    }
  };

  // ---------------- ASK QUESTION ----------------
  const askQuestion = async () => {

    if (!question.trim()) {
      alert("Please enter a question");
      return;
    }

    try {

      setLoading(true);

      // clear previous answer
      setAnswer("");

      const response = await axios.post(
        "http://127.0.0.1:8000/chat",
        {
          question: question
        }
      );

      console.log(response.data);

      setAnswer(
        response.data.answer ||
        response.data.error ||
        "No answer received"
      );

    } catch (error) {

      console.log(error);

      setAnswer(
        error.response?.data?.error ||
        "Question Failed"
      );

    } finally {

      setLoading(false);

    }
  };

  return (

    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#111827",
        color: "white",
        padding: "40px",
        fontFamily: "Arial"
      }}
    >

      <h1 style={{ marginBottom: "30px" }}>
        AI Document Chatbot
      </h1>

      {/* ---------------- Upload Section ---------------- */}

      <div
        style={{
          backgroundColor: "#1F2937",
          padding: "20px",
          borderRadius: "10px",
          marginBottom: "30px"
        }}
      >

        <h2>Upload PDF</h2>

        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <button
          onClick={uploadPDF}
          style={{
            marginLeft: "10px",
            padding: "10px 15px",
            cursor: "pointer"
          }}
        >
          Upload
        </button>

      </div>

      {/* ---------------- Question Section ---------------- */}

      <div
        style={{
          backgroundColor: "#1F2937",
          padding: "20px",
          borderRadius: "10px"
        }}
      >

        <h2>Ask Question</h2>

        <input
          type="text"
          placeholder="Ask question from PDF..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={{
            width: "70%",
            padding: "10px",
            borderRadius: "5px",
            border: "none"
          }}
        />

        <button
          onClick={askQuestion}
          style={{
            marginLeft: "10px",
            padding: "10px 15px",
            cursor: "pointer"
          }}
        >
          Ask
        </button>

      </div>

      {/* ---------------- Loading ---------------- */}

      {
        loading && (
          <p style={{ marginTop: "20px" }}>
            Loading...
          </p>
        )
      }

      {/* ---------------- Answer ---------------- */}

      {
        answer && (
          <div
            style={{
              marginTop: "30px",
              backgroundColor: "#1F2937",
              padding: "20px",
              borderRadius: "10px"
            }}
          >

            <h2>Answer</h2>

            <p
              style={{
                lineHeight: "1.8"
              }}
            >
              {answer}
            </p>

          </div>
        )
      }

    </div>
  );
}

export default App;
