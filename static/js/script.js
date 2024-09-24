document.getElementById('recommendation-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const jobRoleId = document.getElementById('job_role').value;
    const url = `/recommend?job_role_id=${jobRoleId}&top_n=3`;

    fetch(url)
        .then(response => response.json())
        .then(data => {
            const recommendationsDiv = document.getElementById('recommendations');
            recommendationsDiv.innerHTML = '';

            if (data.length === 0) {
                recommendationsDiv.innerHTML = '<p>No recommendations available for this job role.</p>';
                return;
            }

            data.forEach(course => {
                const courseDiv = document.createElement('div');
                courseDiv.className = 'recommendation';
                courseDiv.innerHTML = `
                    <h3>${course.title}</h3>
                    <p>${course.description}</p>
                    <p><strong>Domain:</strong> ${course.domain}</p>
                `;
                recommendationsDiv.appendChild(courseDiv);
            });
        })
        .catch(error => {
            console.error('Error fetching recommendations:', error);
        });
});