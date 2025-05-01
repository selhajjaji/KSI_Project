import { useState } from 'react';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    TRAFFCTL: 'No Control',
    VISIBILITY: 'Clear',
    LIGHT: 'Daylight',
    RDSFCOND: 'Dry',
    IMPACTYPE: 'Approaching',
    INVAGE: 0,
    INVTYPE: 'Passenger',
    Month: 0,
    DayOfWeek: 0,
    Hour: 0,
    SPEEDING: 'No',
    ALCOHOL: 'No',
  });

  const [errors, setErrors] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const validateForm = () => {
    const newErrors = {};

    // Validate numerical fields
    if (formData.INVAGE < 0 || formData.INVAGE > 120) {
      newErrors.INVAGE = 'Age must be between 0 and 120';
    }

    if (formData.Month < 0 || formData.Month > 12) {
      newErrors.Month = 'Month must be between 1 and 12';
    }

    if (formData.DayOfWeek < 0 || formData.DayOfWeek > 7) {
      newErrors.DayOfWeek = 'Day of week must be between 1 and 7';
    }

    if (formData.Hour < 0 || formData.Hour > 23) {
      newErrors.Hour = 'Hour must be between 0 and 23';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: ['INVAGE', 'Month', 'DayOfWeek', 'Hour'].includes(name)
        ? parseFloat(value)
        : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (validateForm()) {
      setLoading(true);
      try {
        const response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(formData),
        });

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const data = await response.json();
        setPrediction(data);
      } catch (error) {
        console.error('Error:', error);
        setPrediction({
          message: 'Error making prediction',
          prediction: -1,
        });
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="container">
      <h1 className="title">Fatality Predictor</h1>
      <form onSubmit={handleSubmit} className="prediction-form">
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="trafficSignal">Traffic Control</label>
            <select
              id="trafficSignal"
              name="TRAFFCTL"
              value={formData.TRAFFCTL}
              onChange={handleChange}
            >
              <option value="No Control">No Control</option>
              <option value="Traffic Signal">Traffic Signal</option>
              <option value="Stop Sign">Stop Sign</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="visibility">Visibility</label>
            <select
              id="visibility"
              name="VISIBILITY"
              value={formData.VISIBILITY}
              onChange={handleChange}
            >
              <option value="Clear">Clear</option>
              <option value="Rain">Rain</option>
              <option value="Fog">Fog</option>
              <option value="Snow">Snow</option>
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="light">Light Condition</label>
            <select
              id="light"
              name="LIGHT"
              value={formData.LIGHT}
              onChange={handleChange}
            >
              <option value="Daylight">Daylight</option>
              <option value="Dark">Dark</option>
              <option value="Dusk">Dusk</option>
              <option value="Dawn">Dawn</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="roadCondition">Road Condition</label>
            <select
              id="roadCondition"
              name="RDSFCOND"
              value={formData.RDSFCOND}
              onChange={handleChange}
            >
              <option value="Dry">Dry</option>
              <option value="Wet">Wet</option>
              <option value="Snow">Snow</option>
              <option value="Ice">Ice</option>
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="impactType">Impact Type</label>
            <select
              id="impactType"
              name="IMPACTYPE"
              value={formData.IMPACTYPE}
              onChange={handleChange}
            >
              <option value="Approaching">Approaching</option>
              <option value="Rear End">Rear End</option>
              <option value="Side Impact">Side Impact</option>
              <option value="Head On">Head On</option>
              <option value="Turning Movement">Turning Movement</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="involvedType">Involved Type</label>
            <select
              id="involvedType"
              name="INVTYPE"
              value={formData.INVTYPE}
              onChange={handleChange}
            >
              <option value="Passenger">Passenger</option>
              <option value="Driver">Driver</option>
              <option value="Pedestrian">Pedestrian</option>
              <option value="Cyclist">Cyclist</option>
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="age">Age</label>
            <input
              type="number"
              id="age"
              name="INVAGE"
              min="0"
              max="120"
              value={formData.INVAGE}
              onChange={handleChange}
            />
            {errors.INVAGE && (
              <span className="error-message">{errors.INVAGE}</span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="month">Month</label>
            <input
              type="number"
              id="month"
              name="Month"
              min="1"
              max="12"
              value={formData.Month}
              onChange={handleChange}
            />
            {errors.Month && (
              <span className="error-message">{errors.Month}</span>
            )}
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="dayOfWeek">Day of Week</label>
            <select
              id="dayOfWeek"
              name="DayOfWeek"
              value={formData.DayOfWeek}
              onChange={handleChange}
            >
              <option value="1">Monday</option>
              <option value="2">Tuesday</option>
              <option value="3">Wednesday</option>
              <option value="4">Thursday</option>
              <option value="5">Friday</option>
              <option value="6">Saturday</option>
              <option value="7">Sunday</option>
            </select>
            {errors.DayOfWeek && (
              <span className="error-message">{errors.DayOfWeek}</span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="hour">Hour (0-23)</label>
            <input
              type="number"
              id="hour"
              name="Hour"
              min="0"
              max="23"
              value={formData.Hour}
              onChange={handleChange}
            />
            {errors.Hour && (
              <span className="error-message">{errors.Hour}</span>
            )}
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Speeding</label>
            <div className="radio-group">
              <label className="radio-option">
                <input
                  type="radio"
                  name="SPEEDING"
                  value="Yes"
                  checked={formData.SPEEDING === 'Yes'}
                  onChange={handleChange}
                />
                Yes
              </label>
              <label className="radio-option">
                <input
                  type="radio"
                  name="SPEEDING"
                  value="No"
                  checked={formData.SPEEDING === 'No'}
                  onChange={handleChange}
                />
                No
              </label>
            </div>
          </div>

          <div className="form-group">
            <label>Alcohol</label>
            <div className="radio-group">
              <label className="radio-option">
                <input
                  type="radio"
                  name="ALCOHOL"
                  value="Yes"
                  checked={formData.ALCOHOL === 'Yes'}
                  onChange={handleChange}
                />
                Yes
              </label>
              <label className="radio-option">
                <input
                  type="radio"
                  name="ALCOHOL"
                  value="No"
                  checked={formData.ALCOHOL === 'No'}
                  onChange={handleChange}
                />
                No
              </label>
            </div>
          </div>
        </div>

        <button type="submit" className="submit-button" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict Fatality'}
        </button>

        {prediction && (
          <div className="prediction-result">
            <h2>Prediction Result</h2>
            <div
              className={`result-message ${prediction.message.toLowerCase()}`}
            >
              {prediction.message}
            </div>
          </div>
        )}
      </form>
    </div>
  );
}

export default App;
