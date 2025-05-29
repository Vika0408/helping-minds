import React, { useState, useEffect } from "react";

const DiaryPage = () => {
  const [entry, setEntry] = useState("");
  const [entries, setEntries] = useState([]);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split("T")[0]);

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem("diaryEntries") || "[]");
    setEntries(saved);
    const found = saved.find((e) => e.date === selectedDate);
    setEntry(found ? found.content : "");
  }, [selectedDate]);

  const handleSave = () => {
    const updatedEntries = entries.filter((e) => e.date !== selectedDate);
    updatedEntries.push({ date: selectedDate, content: entry });
    localStorage.setItem("diaryEntries", JSON.stringify(updatedEntries));
    setEntries(updatedEntries);
    alert("Entry saved for " + selectedDate);
  };

  const handleDelete = (dateToDelete) => {
    const updatedEntries = entries.filter((e) => e.date !== dateToDelete);
    localStorage.setItem("diaryEntries", JSON.stringify(updatedEntries));
    setEntries(updatedEntries);

    // Clear editor if current selected date is deleted
    if (selectedDate === dateToDelete) {
      setSelectedDate(new Date().toISOString().split("T")[0]);
      setEntry("");
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 mt-10">
      <h1 className="text-4xl font-bold text-blue-700 mb-6 text-center">📖 My Personal Diary</h1>

      <div className="grid md:grid-cols-3 gap-6">
        {/* Entry List */}
        <div className="bg-white rounded-xl shadow-md p-4 max-h-[500px] overflow-y-auto">
          <h2 className="text-xl font-semibold text-gray-700 mb-4">Previous Entries</h2>
          {entries
            .sort((a, b) => new Date(b.date) - new Date(a.date))
            .map((e, i) => (
              <div
                key={i}
                className={`flex justify-between items-center p-2 rounded cursor-pointer hover:bg-blue-100 ${
                  e.date === selectedDate ? "bg-blue-200" : ""
                }`}
              >
                <span onClick={() => setSelectedDate(e.date)}>
                  {new Date(e.date).toDateString()}
                </span>
                <button
                  onClick={() => handleDelete(e.date)}
                  className="text-red-500 hover:text-red-700 ml-4"
                  title="Delete entry"
                >
                  🗑️
                </button>
              </div>
            ))}
        </div>

        {/* Editor Section */}
        <div className="md:col-span-2 bg-white rounded-xl shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold text-gray-800">Write for: </h2>
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="border px-3 py-1 rounded text-gray-700"
            />
          </div>

          <textarea
            value={entry}
            onChange={(e) => setEntry(e.target.value)}
            rows="15"
            placeholder="Start writing your thoughts here..."
            className="w-full border rounded-lg p-4 text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
          />

          <button
            onClick={handleSave}
            className="mt-4 bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
          >
            Save Entry
          </button>
        </div>
      </div>
    </div>
  );
};

export default DiaryPage;