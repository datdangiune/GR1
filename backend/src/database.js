const { Sequelize } = require('sequelize');

const sequelize = new Sequelize('postgresql://neondb_owner:npg_upTdhI70yUwS@ep-rapid-haze-a19dd3tm-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require', {
    dialect: 'postgres',
    dialectOptions: {
        ssl: {
            require: true,
            rejectUnauthorized: false, 
        },
    },
});

sequelize.authenticate()
    .then(() => console.log('Database connected successfully.'))
    .catch((err) => console.error('Unable to connect to the database:', err));

module.exports = sequelize;